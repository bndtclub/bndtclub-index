# Define default environment variables
INFO=true

# Read the .env file and export variables using xargs
export_env:
	export $(cat .env | xargs -n 1) && echo "Environment variables exported."

# Default target
.PHONY: run

# Normal mode (INFO only)
run: export_env
	@INFO=$(INFO) streamlit run app.py

# Debug mode (INFO and DEBUG)
debug: export_env
	@DEBUG=true INFO=$(INFO) streamlit run app.py
