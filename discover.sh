#!/usr/bin/env bash
for id in $(docker ps -q); do
    mounts=$(docker inspect "$id" --format '{{range .Mounts}}{{.Source}}{{"\n"}}{{end}}' 2>/dev/null | grep "it-share")
    if [ -n "$mounts" ]; then
        name=$(docker inspect "$id" --format '{{.Name}}' | sed 's|/||')
        echo "✓ $id  ($name)"
        echo "$mounts"
        echo
    fi
done
