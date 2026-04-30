param(
    [string]$EnvName = "tf-pwa-env",
    [string]$PythonVersion = "3.9",
    [string]$PackagePath = "tf-pwa"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$packageFullPath = Join-Path $projectRoot $PackagePath

function Invoke-CondaChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [string]$FailureMessage = "Conda command failed."
    )

    & conda @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw $FailureMessage
    }
}

function Invoke-CondaRunChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$EnvironmentName,
        [Parameter(Mandatory = $true)]
        [string[]]$CommandArguments,
        [string]$FailureMessage = "conda run command failed."
    )

    & conda run -n $EnvironmentName @CommandArguments
    if ($LASTEXITCODE -ne 0) {
        throw $FailureMessage
    }
}

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "conda was not found in PATH. Open a Conda-enabled PowerShell first, or run 'conda init powershell' and restart the terminal."
}

if (-not (Test-Path $packageFullPath)) {
    throw "Local tf-pwa path not found: $packageFullPath"
}

if (-not (Test-Path (Join-Path $packageFullPath "setup.py")) -and -not (Test-Path (Join-Path $packageFullPath "pyproject.toml"))) {
    throw "Local tf-pwa checkout at '$packageFullPath' does not look installable."
}

Write-Host "Project root: $projectRoot"
Write-Host "Creating Conda environment '$EnvName' with Python $PythonVersion ..."
Invoke-CondaChecked `
    -Arguments @("create", "-y", "-n", $EnvName, "--override-channels", "-c", "conda-forge", "python=$PythonVersion", "pip") `
    -FailureMessage @"
Failed to create the Conda environment '$EnvName'.
This script now uses conda-forge only. If the error mentions connectivity,
please check whether your network can reach https://conda.anaconda.org/conda-forge .
"@

Write-Host "Installing Conda packages into '$EnvName' ..."
Invoke-CondaChecked `
    -Arguments @(
        "install", "-y", "-n", $EnvName, "-c", "conda-forge",
        "numpy<1.25.0", "scipy", "matplotlib", "jupyterlab", "ipython",
        "pandas", "tqdm", "h5py", "numba", "git", "pip"
    ) `
    -FailureMessage "Failed to install the Conda packages into '$EnvName'."

Write-Host "Upgrading pip/setuptools/wheel in '$EnvName' ..."
Invoke-CondaRunChecked `
    -EnvironmentName $EnvName `
    -CommandArguments @("python", "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel") `
    -FailureMessage "Failed to upgrade pip/setuptools/wheel in '$EnvName'."

Write-Host "Installing TensorFlow and Python dependencies into '$EnvName' ..."
Invoke-CondaRunChecked `
    -EnvironmentName $EnvName `
    -CommandArguments @(
        "python", "-m", "pip", "install",
        "tensorflow>=2.7", "tensorflow-probability", "pyyaml", "graphviz"
    ) `
    -FailureMessage "Failed to install TensorFlow and Python dependencies into '$EnvName'."

Write-Host "Installing local tf-pwa checkout in editable mode from '$packageFullPath' ..."
Invoke-CondaRunChecked `
    -EnvironmentName $EnvName `
    -CommandArguments @("python", "-m", "pip", "install", "-e", $packageFullPath) `
    -FailureMessage "Failed to install the local tf-pwa checkout into '$EnvName'."

Write-Host ""
Write-Host "Environment '$EnvName' is ready."
Write-Host "Activate it with:"
Write-Host "  conda activate $EnvName"
