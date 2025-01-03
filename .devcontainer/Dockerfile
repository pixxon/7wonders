FROM ubuntu:24.04

RUN apt update && \
    apt install -y --no-install-recommends \
    git \
    zsh \
    wget \
    gnupg2 \
    openssh-client \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

USER ubuntu

RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh && \
    ZSH_PLUGINS=$HOME/.oh-my-zsh/custom/plugins && \
    git clone --single-branch --branch '0.8.0' --depth 1 https://github.com/zsh-users/zsh-syntax-highlighting.git $ZSH_PLUGINS/zsh-syntax-highlighting && \
    git clone --single-branch --branch 'v0.7.0' --depth 1 https://github.com/zsh-users/zsh-autosuggestions.git $ZSH_PLUGINS/zsh-autosuggestions && \
    git clone --single-branch --branch '23.07.13' --depth 1 https://github.com/marlonrichert/zsh-autocomplete.git $ZSH_PLUGINS/zsh-autocomplete && \
    sed -i 's/plugins=(git)/plugins=(git zsh-autosuggestions zsh-syntax-highlighting zsh-autocomplete)/g' $HOME/.zshrc

USER root

RUN apt update && \
    apt install -y --no-install-recommends \
    python3 \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

USER ubuntu

ENTRYPOINT [ "zsh" ]
