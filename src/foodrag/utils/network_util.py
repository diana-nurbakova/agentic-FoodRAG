import os
import platform
import subprocess


def get_wsl_ip():
    try:
        # Run the command to get the WSL IP address
        result = subprocess.run(['wsl', 'hostname', '-I'], capture_output=True, text=True)
        # The IP address is in the stdout of the result
        ip_address = result.stdout.strip()
        return ip_address
    except Exception as e:
        print(f"Error retrieving WSL IP address: {e}")
        return None


def set_nebula_env_vars():
    # Set the NebulaGraph connection details
    port = 9669
    user = 'root'
    password = 'nebula'

    # Determine the IP address to use
    if platform.system() == "Linux":
        # Use localhost for Linux systems
        ip_address = "localhost"
    else:
        # Use the WSL IP for Windows systems
        ip_address = get_wsl_ip()
        if not ip_address:
            raise RuntimeError("Failed to retrieve WSL IP address")

    # Set environment variables for NebulaGraph connection
    os.environ["NEBULA_USER"] = user
    os.environ["NEBULA_PASSWORD"] = password
    os.environ["NEBULA_ADDRESS"] = f"{ip_address}:{port}"

    return ip_address, port, user, password
