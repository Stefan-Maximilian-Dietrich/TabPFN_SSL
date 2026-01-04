#!/bin/bash

USER_NAME=$(whoami)

echo "Jobs von Benutzer: $USER_NAME"
echo "Lösche alle Jobs, deren Name NICHT mit 'sys/' beginnt"
echo

# Jobs anzeigen, die gelöscht werden
squeue -u "$USER_NAME" -h -o "%i %j" | awk '$2 !~ /^sys\// {print}'

echo
read -p "Diese Jobs wirklich löschen? (yes/no): " CONFIRM

if [[ "$CONFIRM" != "yes" ]]; then
    echo "Abgebrochen."
    exit 0
fi

# Jobs löschen
squeue -u "$USER_NAME" -h -o "%i %j" | \
awk '$2 !~ /^sys\// {print $1}' | \
xargs -r scancel

echo "Fertig. Alle nicht-sys/ Jobs wurden gelöscht."

sleep 10

squeue -u di35lox