import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

from .tools import SimulationTools
from .paths import RUNS


INSTRUCTIONS = (
    "你通过工具控制 MuJoCo 中的 OpenArm。先阅读头部相机图像，结合用户任务选择桌面上的像素目标。"
    "图像原点在左上角，u 向右、v 向下；图像上显示像素坐标网格。"
    "reach_pixel 根据已知桌面平面计算目标，TCP 保持在桌面上方 0.14 米。"
    "右臂用于机器人右侧工作区，左臂用于左侧工作区。"
    "必须使用最近 observe 返回的 frame_id。一次动作完成后重新 observe，依据工具返回的实际状态决定后续动作。"
    "仅在工具返回 succeeded 时说明对应动作完成。失败时依据明确原因重新选择可达目标。"
    "当前工具支持相机观测和 IK reach，回答需要准确描述已经执行的动作。"
)


def run_agent(client: OpenAI, model: str, instruction: str, tools: SimulationTools, max_turns: int, trace: Path) -> str:
    observation = tools.call("observe", {})
    messages = [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": [
            {"type": "text", "text": instruction + "\n当前观测：" + json.dumps(observation, ensure_ascii=False)},
            tools.image_content(),
        ]},
    ]
    trace.parent.mkdir(parents=True, exist_ok=True)
    with trace.open("w") as output:
        for turn in range(max_turns):
            response = client.chat.completions.create(
                model=model, messages=messages, tools=tools.schemas(),
                parallel_tool_calls=False,
            )
            message = response.choices[0].message
            output.write(json.dumps({"turn": turn, "message": message.model_dump(mode="json")}, ensure_ascii=False) + "\n")
            output.flush()
            if not message.tool_calls:
                if not message.content:
                    raise RuntimeError("VLM 返回了空内容")
                return message.content
            messages.append({
                "role": "assistant", "content": message.content,
                "tool_calls": [
                    {"id": call.id, "type": "function", "function": {"name": call.function.name, "arguments": call.function.arguments}}
                    for call in message.tool_calls
                ],
            })
            for call in message.tool_calls:
                result = tools.call(call.function.name, json.loads(call.function.arguments))
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False)})
                output.write(json.dumps({"tool": call.function.name, "arguments": json.loads(call.function.arguments), "result": result}, ensure_ascii=False) + "\n")
                output.flush()
            observation = tools.call("observe", {})
            messages.append({"role": "user", "content": [
                {"type": "text", "text": "执行后的最新观测：" + json.dumps(observation, ensure_ascii=False)},
                tools.image_content(),
            ]})
    raise RuntimeError("VLM 工具调用达到 max-turns 限制，任务尚未结束")


def main() -> None:
    parser = argparse.ArgumentParser(description="通过 OpenAI-compatible VLM 调用 OpenArm 仿真工具")
    parser.add_argument("instruction")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--planner", choices=["curobo", "mink"], default="curobo")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--trace", type=Path, default=RUNS / "agent/trace.jsonl")
    args = parser.parse_args()
    if args.max_turns <= 0:
        raise ValueError("max-turns 需要为正整数")
    api_key = os.environ["OPENARM_API_KEY"]
    with OpenAI(api_key=api_key, base_url=args.base_url, max_retries=0, timeout=60) as client:
        with SimulationTools(planner=args.planner) as tools:
            print(run_agent(client, args.model, args.instruction, tools, args.max_turns, args.trace))


if __name__ == "__main__":
    main()
