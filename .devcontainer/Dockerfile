# agent monorepo dev container
# 多言語サポート（Bun / Go / Python + uv）+ GitHub CLI、非 root ユーザー前提

FROM ubuntu:24.04

ARG BUN_VERSION=1.3.11
ARG GO_VERSION=1.23.4
ARG PYTHON_VERSION=3.12
ARG USERNAME=vscode

# 必須ツール
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    sudo \
    build-essential \
    unzip \
    locales \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    && locale-gen en_US.UTF-8 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        | tee /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y --no-install-recommends gh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Go（公式 tarball、バージョン pinning）
RUN ARCH=$(dpkg --print-architecture) \
    && curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-${ARCH}.tar.gz" | tar -C /usr/local -xz
ENV PATH="/usr/local/go/bin:${PATH}"

# 非 root ユーザー
RUN useradd -ms /bin/bash ${USERNAME} \
    && usermod -aG sudo ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

USER ${USERNAME}
WORKDIR /workspaces/agent

# Bun（ユーザー領域にインストール、バージョン pinning）
RUN curl -fsSL "https://bun.sh/install" | bash -s "bun-v${BUN_VERSION}"
ENV BUN_INSTALL=/home/${USERNAME}/.bun
ENV PATH="${BUN_INSTALL}/bin:${PATH}"

# uv（ユーザー領域にインストール）
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

ENV DEVCONTAINER=true

CMD ["bash"]
