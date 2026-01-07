#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# 1) micromamba lokal installieren
# ----------------------------
mkdir -p "$HOME/bin"

if [ ! -f "$HOME/bin/micromamba" ]; then
  echo "[setup] installing micromamba to $HOME/bin/micromamba"
  cd "$HOME"
  curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
  mv bin/micromamba "$HOME/bin/micromamba"
  rm -rf bin
else
  echo "[setup] micromamba already exists at $HOME/bin/micromamba"
fi

# PATH dauerhaft setzen (falls noch nicht drin)
if ! grep -q 'export PATH="$HOME/bin:$PATH"' "$HOME/.bashrc" 2>/dev/null; then
  echo 'export PATH="$HOME/bin:$PATH"' >> "$HOME/.bashrc"
  echo "[setup] added $HOME/bin to PATH in ~/.bashrc"
fi

export PATH="$HOME/bin:$PATH"

# ----------------------------
# 2) R-Environment erstellen
# ----------------------------
echo "[setup] creating micromamba env: r_eval"
"$HOME/bin/micromamba" create -y -n r_eval -c conda-forge r-base r-data.table r-ggplot2

echo "[setup] done."
echo ""
echo "Jetzt bitte in deiner Shell einmal ausführen:"
echo "  source ~/.bashrc"
echo "  micromamba activate r_eval"
echo "  which Rscript"
echo "  Rscript --version"
