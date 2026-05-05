#!/usr/bin/env bash

quicksetup_env_dir() {
    cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

quicksetup_env_file() {
    echo "$(quicksetup_env_dir)/.env"
}

load_quicksetup_env() {
    local env_file
    env_file="$(quicksetup_env_file)"

    if [ ! -f "$env_file" ]; then
        cp "$(quicksetup_env_dir)/.env.example" "$env_file"
        chmod 600 "$env_file" 2>/dev/null || true
    fi

    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
}

save_quicksetup_env_var() {
    local key="$1"
    local value="$2"
    local env_file tmp_file escaped

    env_file="$(quicksetup_env_file)"
    tmp_file="${env_file}.tmp"
    printf -v escaped "%q" "$value"

    touch "$env_file"
    chmod 600 "$env_file" 2>/dev/null || true

    if grep -q "^${key}=" "$env_file"; then
        awk -v key="$key" -v value="${key}=${escaped}" '
            BEGIN { replaced = 0 }
            $0 ~ "^" key "=" {
                print value
                replaced = 1
                next
            }
            { print }
            END {
                if (!replaced) {
                    print value
                }
            }
        ' "$env_file" > "$tmp_file"
    else
        cp "$env_file" "$tmp_file"
        printf "%s=%s\n" "$key" "$escaped" >> "$tmp_file"
    fi

    mv "$tmp_file" "$env_file"
    chmod 600 "$env_file" 2>/dev/null || true
    export "$key=$value"
}

prompt_secret() {
    local key="$1"
    local prompt="$2"
    local value

    read -r -s -p "$prompt: " value
    echo
    save_quicksetup_env_var "$key" "$value"
}

prompt_value() {
    local key="$1"
    local prompt="$2"
    local value

    read -r -p "$prompt: " value
    save_quicksetup_env_var "$key" "$value"
}

github_token_is_valid() {
    local token="$1"
    local repo="${2:-masiarhub/robot-learning-rl-project}"

    [ -n "$token" ] || return 1

    if command -v curl &>/dev/null; then
        curl -fsS \
            -H "Authorization: Bearer ${token}" \
            -H "Accept: application/vnd.github+json" \
            "https://api.github.com/repos/${repo}" >/dev/null
    else
        git ls-remote "https://${token}@github.com/${repo}.git" HEAD &>/dev/null
    fi
}

ensure_github_token() {
    local purpose="${1:-GitHub access}"
    local repo="${2:-masiarhub/robot-learning-rl-project}"

    if ! github_token_is_valid "${GITHUB_TOKEN:-}" "$repo"; then
        echo ""
        echo "GitHub token missing or invalid for ${purpose}."
        echo "Create one at https://github.com/settings/tokens with repo scope."
        prompt_secret "GITHUB_TOKEN" "GitHub token"
        if ! github_token_is_valid "${GITHUB_TOKEN:-}" "$repo"; then
            echo "  x GitHub token is still invalid for ${repo}."
            exit 1
        fi
    fi
}

hf_token_is_valid() {
    local token="$1"

    [ -n "$token" ] || return 1

    if command -v curl &>/dev/null; then
        curl -fsS \
            -H "Authorization: Bearer ${token}" \
            "https://huggingface.co/api/whoami-v2" >/dev/null
    elif command -v hf &>/dev/null; then
        HF_TOKEN="$token" hf auth whoami &>/dev/null
    else
        return 0
    fi
}

ensure_hf_token() {
    if ! hf_token_is_valid "${HF_TOKEN:-}"; then
        echo ""
        echo "Hugging Face token missing or invalid."
        echo "Create a write-access token at https://huggingface.co/settings/tokens."
        prompt_secret "HF_TOKEN" "Hugging Face token"
        if ! hf_token_is_valid "${HF_TOKEN:-}"; then
            echo "  x Hugging Face token is still invalid."
            exit 1
        fi
    fi
}

ensure_hf_repo_prefix() {
    if [ -z "${HF_REPO_PREFIX:-}" ]; then
        echo ""
        prompt_value "HF_REPO_PREFIX" "Hugging Face username/org prefix for model repos (e.g. pcwagner)"
    fi
}
