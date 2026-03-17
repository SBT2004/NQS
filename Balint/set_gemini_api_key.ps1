param(
    [string]$Key
)

if (-not $Key) {
    $secure = Read-Host -Prompt "Enter GEMINI_API_KEY" -AsSecureString
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try {
        $Key = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }
}

if (-not $Key) {
    throw "GEMINI_API_KEY was empty."
}

[Environment]::SetEnvironmentVariable("GEMINI_API_KEY", $Key, "User")
$env:GEMINI_API_KEY = $Key

Write-Host "Set GEMINI_API_KEY for the current session and the current Windows user."
Write-Host "Open a new terminal or restart Codex/PyCharm if they were already running."
