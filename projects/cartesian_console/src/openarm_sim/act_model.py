from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18


@dataclass
class ACTConfig:
    state_dim: int = 57
    action_dim: int = 8
    chunk: int = 20
    hidden: int = 128
    heads: int = 4
    layers: int = 2
    latent: int = 16
    vision: bool = False
    relative_commands: bool = False
    physical_oracle: bool = False
    observation_clip: float = 1000000.0
    image_width: int = 192
    image_height: int = 108
    image_feature_stage: int = 4


class ActionChunkTransformer(nn.Module):
    def __init__(self, config: ACTConfig, state_mean, state_std, action_mean, action_std):
        super().__init__()
        self.config = config
        for name, value in (("state_mean", state_mean), ("state_std", state_std),
                            ("action_mean", action_mean), ("action_std", action_std)):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        if self.state_mean.shape != (config.state_dim,) or self.action_mean.shape != (config.action_dim,):
            raise ValueError("ACT normalization dimensions do not match the model")
        if bool((self.state_std <= 0).any()) or bool((self.action_std <= 0).any()):
            raise ValueError("ACT normalization scales must be positive")
        if config.physical_oracle and (config.state_dim != 57 or config.vision):
            raise ValueError("Physical oracle features require the 57-dimensional teacher state")
        indices = [*range(7), 14, *range(16, 23), *range(29, 34), 51, *range(54, 57)] if config.physical_oracle else list(range(config.state_dim))
        self.register_buffer("state_indices", torch.tensor(indices, dtype=torch.long))
        self.state_projection = nn.Linear(len(indices), config.hidden)
        self.action_projection = nn.Linear(config.action_dim, config.hidden)
        self.cls = nn.Parameter(torch.zeros(1, 1, config.hidden))
        self.posterior_positions = nn.Parameter(torch.randn(1, config.chunk + 2, config.hidden) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(config.hidden, config.heads, config.hidden * 4,
                                                   dropout=0.0, batch_first=True, norm_first=True)
        self.posterior = nn.TransformerEncoder(encoder_layer, config.layers, enable_nested_tensor=False)
        self.latent_parameters = nn.Linear(config.hidden, 2 * config.latent)
        self.latent_projection = nn.Linear(config.latent, config.hidden)
        self.memory_positions = nn.Parameter(torch.randn(1, 2, config.hidden) * 0.02)
        self.queries = nn.Parameter(torch.randn(1, config.chunk, config.hidden) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(config.hidden, config.heads, config.hidden * 4,
                                                   dropout=0.0, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, config.layers, norm=nn.LayerNorm(config.hidden))
        self.action_head = nn.Linear(config.hidden, config.action_dim)
        if config.vision:
            if config.image_feature_stage not in (3, 4):
                raise ValueError("Visual feature stage must be 3 or 4")
            backbone = resnet18(weights=None)
            self.backbone = nn.Sequential(*list(backbone.children())[:-(6 - config.image_feature_stage)])
            channels = 256 if config.image_feature_stage == 3 else 512
            self.image_projection = nn.Conv2d(channels, config.hidden, 1)
            self.image_positions = nn.Parameter(torch.randn(1, 256 if config.image_feature_stage == 3 else 64, config.hidden) * 0.02)
            self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
            self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

    def forward(self, state, actions=None, padding=None, images=None):
        raw_state = state
        state = torch.clamp((state - self.state_mean) / self.state_std,
                            -self.config.observation_clip, self.config.observation_clip)[:, self.state_indices]
        batch = len(state)
        state_token = self.state_projection(state)[:, None]
        mean = logvar = None
        if actions is not None:
            if actions.shape[1:] != (self.config.chunk, self.config.action_dim) or padding.shape != actions.shape[:2]:
                raise ValueError("ACT action chunk or padding dimensions are invalid")
            tokens = torch.cat((self.cls.expand(batch, -1, -1), state_token,
                                self.action_projection(self.normalized_actions(actions, raw_state))), dim=1)
            mask = torch.cat((torch.zeros((batch, 2), dtype=torch.bool, device=state.device), padding), dim=1)
            encoded = self.posterior(tokens + self.posterior_positions, src_key_padding_mask=mask)[:, 0]
            mean, logvar = self.latent_parameters(encoded).chunk(2, dim=-1)
            logvar = logvar.clamp(-10, 10)
            latent = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean) if self.training else mean
        else:
            latent = torch.zeros((batch, self.config.latent), device=state.device)
        memory = torch.cat((state_token, self.latent_projection(latent)[:, None]), dim=1) + self.memory_positions
        if self.config.vision:
            if images is None or images.shape[1:] != (3, self.config.image_height, self.config.image_width):
                raise ValueError("Visual ACT requires RGB observations")
            features = self.image_projection(self.backbone((images - self.image_mean) / self.image_std))
            image_tokens = features.flatten(2).transpose(1, 2)
            if image_tokens.shape[1] > self.image_positions.shape[1]:
                raise ValueError("Image resolution exceeds the configured ACT image positions")
            memory = torch.cat((memory, image_tokens + self.image_positions[:, :image_tokens.shape[1]]), dim=1)
        decoded = self.decoder(self.queries.expand(batch, -1, -1), memory)
        prediction = self.action_head(decoded)
        return prediction, mean, logvar

    def loss(self, state, actions, padding, images=None, kl_weight: float = 1.0):
        prediction, mean, logvar = self(state, actions, padding, images)
        target = self.normalized_actions(actions, state)
        valid = (~padding).unsqueeze(-1)
        reconstruction = ((prediction - target).abs() * valid).sum() / (valid.sum() * self.config.action_dim)
        kl = (-0.5 * (1 + logvar - mean.square() - logvar.exp())).sum(dim=-1).mean()
        return reconstruction + kl_weight * kl, reconstruction, kl

    @torch.no_grad()
    def commands(self, state, images=None):
        prediction, _, _ = self(state, images=images)
        command = prediction * self.action_std + self.action_mean
        if self.config.relative_commands:
            command = command + self.command_reference(state)[:, None]
        command[..., 7].clamp_(0, 0.044)
        return command

    def normalized_actions(self, actions, state):
        if self.config.relative_commands:
            actions = actions - self.command_reference(state)[:, None]
        return (actions - self.action_mean) / self.action_std

    def command_reference(self, state):
        return state[:, :8] if self.config.state_dim == 9 else torch.cat((state[:, :7], state[:, 14:15]), dim=1)

    def distillation_loss(self, state, actions, padding, images):
        prediction, _, _ = self(state, images=images)
        target = self.normalized_actions(actions, state)
        valid = (~padding).unsqueeze(-1)
        loss = ((prediction - target).abs() * valid).sum() / (valid.sum() * self.config.action_dim)
        return loss, loss, loss.new_zeros(())

    def save(self, path: Path, **metadata):
        temporary = path.with_suffix(path.suffix + ".partial")
        torch.save({"schema": "openarm.act.v1", "config": asdict(self.config),
                    "model": self.state_dict(), "metadata": metadata}, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path, device="cuda"):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if checkpoint["schema"] != "openarm.act.v1":
            raise ValueError("Unsupported ACT checkpoint")
        state = checkpoint["model"]
        model = cls(ACTConfig(**checkpoint["config"]), state["state_mean"], state["state_std"],
                    state["action_mean"], state["action_std"]).to(device)
        if "state_indices" not in state:
            state["state_indices"] = model.state_indices
        model.load_state_dict(state)
        model.eval()
        return model


def normalization(dataset, relative_commands=False):
    action = dataset.command
    if relative_commands:
        reference = dataset.state[:, :8] if dataset.state.shape[1] == 9 else np.concatenate((dataset.state[:, :7], dataset.state[:, 14:15]), axis=1)
        action = action - reference
    return (dataset.state.mean(axis=0), np.maximum(dataset.state.std(axis=0), 0.01),
            action.mean(axis=0), np.maximum(action.std(axis=0), 0.001))


class ACTController:
    def __init__(self, model: ActionChunkTransformer, worlds: int, query_interval: int = 5):
        if not 1 <= query_interval <= model.config.chunk:
            raise ValueError("ACT query interval must lie within the predicted chunk")
        self.model, self.query_interval = model, query_interval
        self.device = next(model.parameters()).device
        self.chunk = torch.zeros((worlds, model.config.chunk, 8), device=self.device)
        self.phase = torch.zeros(worlds, dtype=torch.long, device=self.device)

    def reset(self, indices):
        self.phase[indices] = 0

    @torch.no_grad()
    def command(self, observation, images=None):
        observation = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        selected = self.phase == 0
        if bool(selected.any()):
            self.chunk[selected] = self.model.commands(observation[selected], images[selected] if images is not None else None)
        command = self.chunk[torch.arange(len(self.phase), device=self.device), self.phase].clone()
        self.phase = (self.phase + 1) % self.query_interval
        return command
