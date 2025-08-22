if [ "$(readlink ~/.ssh/agent.sock)" = "$SSH_AUTH_SOCK" ]; then
  echo "Symlink zeigt auf den richtigen Socket."
else
  echo "Symlink zeigt woanders hin oder existiert nicht."
fi
