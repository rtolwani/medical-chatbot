#!/bin/bash

# Create references directory in tmp (for production)
mkdir -p /tmp/references

# Copy PDF files to tmp directory if in production
if [ "$FLASK_ENV" = "production" ]; then
    cp -r references/*.pdf /tmp/references/ 2>/dev/null || true
fi

# Default port if not set
export PORT=${PORT:-8080}

# Start the application with worker timeout
exec gunicorn "app:create_app()" \
    --bind "0.0.0.0:$PORT" \
    --workers 2 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile -
