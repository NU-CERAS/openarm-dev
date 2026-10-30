# Run from PowerShell. MuJoCo and ROS both run inside Ubuntu WSL, using WSLg.
[CmdletBinding()]
param(
    [string]$Distro = 'Ubuntu-24.04',
    [string]$RepoPath = '~/dev/openarm-dev',
    [switch]$BuildRos,
    [switch]$Hardware,
    [switch]$Console,
    [string]$Branch
)
$ErrorActionPreference = 'Stop'
function Quote-Sh([string]$Value) {
    $Quote = [string][char]39
    $EscapedQuote = $Quote + '"' + $Quote + '"' + $Quote
    return $Quote + $Value.Replace($Quote, $EscapedQuote) + $Quote
}
# Existing repositories are reused. A new WSL checkout avoids /mnt/c build issues.
$RepoRoot = Split-Path $PSScriptRoot -Parent
$Remote = & git -C $RepoRoot remote get-url origin
if ($LASTEXITCODE -ne 0) { throw 'Set a reachable git origin before bootstrapping a new WSL checkout.' }
if (-not $Branch) {
    $Branch = & git -C $RepoRoot symbolic-ref --quiet --short HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Specify -Branch when starting from a detached checkout.' }
    $Branch = $Branch.Trim()
}
$Distros = (& wsl.exe --list --quiet) -replace "`0", ''
if ($LASTEXITCODE -ne 0 -or $Distro -notin $Distros.Trim()) {
    & wsl.exe --install -d $Distro
    if ($LASTEXITCODE -ne 0) { throw 'WSL installation failed. Run PowerShell as administrator if requested.' }
    Write-Host 'Finish Ubuntu first-run user setup (and reboot if requested), then rerun this script.'
    exit 0
}
$Flags = ''
if ($BuildRos) { $Flags += ' --build-ros' }
if ($Hardware) { $Flags += ' --hardware' }
if ($Console) { $Flags += ' --console' }
$Destination = Quote-Sh $RepoPath
$Origin = Quote-Sh $Remote.Trim()
$BranchName = Quote-Sh $Branch
$Command = @'
set -eu
repo=REPO_DESTINATION
case "$repo" in '~/'*) repo="$HOME/${repo#\~/}" ;; esac
if [ ! -d "$repo/.git" ]; then
  command -v git >/dev/null || { sudo apt-get update && sudo apt-get install -y git; }
  mkdir -p "$(dirname "$repo")"
  git clone --recursive --branch REMOTE_BRANCH REMOTE_ORIGIN "$repo"
fi
cd "$repo"
[ -f setup/linux.sh ] || { echo 'This WSL checkout does not contain the new setup scripts. Copy/push your local changes first.' >&2; exit 1; }
bash setup/linux.sh SETUP_FLAGS
'@
$Command = $Command.Replace('REPO_DESTINATION', $Destination).Replace('REMOTE_ORIGIN', $Origin).Replace('REMOTE_BRANCH', $BranchName).Replace('SETUP_FLAGS', $Flags)
& wsl.exe -d $Distro -- bash -lc $Command
if ($LASTEXITCODE -ne 0) { throw 'Ubuntu setup failed; see the output above.' }
Write-Host "Ready in $Distro at $RepoPath. This is a separate checkout; uncommitted Windows changes are not copied."
