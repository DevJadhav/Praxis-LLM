#!/bin/sh

# Install pip if not already available
python -m ensurepip --upgrade || true

# Print Bytewax version for debugging
uv run -m pip show bytewax | grep Version

# Set RUST_BACKTRACE for better error messages
export RUST_BACKTRACE=full

if [ "$DEBUG" = true ]
then
    uv run -m bytewax.run "tools.run_real_time:build_flow(debug=True)"
else
    if [ "$BYTEWAX_PYTHON_FILE_PATH" = "" ]
    then
        echo 'BYTEWAX_PYTHON_FILE_PATH is not set. Exiting...'
        exit 1
    fi
    echo "Running with BYTEWAX_PYTHON_FILE_PATH=$BYTEWAX_PYTHON_FILE_PATH"
    uv run -m bytewax.run "$BYTEWAX_PYTHON_FILE_PATH"
fi


echo 'Process ended.'

if [ "$BYTEWAX_KEEP_CONTAINER_ALIVE" = true ]
then
    echo 'Keeping container alive...';
    while :; do sleep 1; done
fi