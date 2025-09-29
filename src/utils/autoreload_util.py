import os


# This method is used to automatically 'aimport' all Python modules in the given project for IPython autoreload for
# debugging purposes.
# The following must be set at the ipython_config.py file in the startup directory of the IPython profile:
# autoreload_project_path = 'D:/DataspellProjects/foodRAG/'  # Path to the project
# autoreload_util_path = 'src.utils.autoreload_util'  # Import-Path to the autoreload_util.py from the project directory
# c.InteractiveShellApp.extensions = ['autoreload']
# c.InteractiveShellApp.exec_lines = [
#     '%autoreload 1',  # Enable autoreload extension
#     f'import sys; sys.path.append("{autoreload_project_path}")',  # Add project path to sys.path
#     f'from {autoreload_util_path} import autoreload_all_modules; autoreload_all_modules("{autoreload_project_path}")'
#     # Call the autoreload function directly after importing
# ]
# Also refer to https://stackoverflow.com/questions/5364050/reloading-submodules-in-ipython
def autoreload_all_modules(project_path: str):
    """
    Automatically aimport all Python modules in the given project for IPython autoreload.
    :param project_path: The root package path of the project.
    """
    print(f"Enable Auto-reloading all modules in project {project_path}...")
    try:
        import IPython
    except ImportError:
        raise ImportError("IPython is required to use this autoreload function.")

    ipython = IPython.get_ipython()
    if not ipython:
        raise RuntimeError("This function must be called from within an IPython session.")

    if not project_path:
        raise ValueError(f"Project {project_path} not found in sys.path. Ensure it is added to sys.path.")

    # Find all modules in the project
    for root, _, files in os.walk(project_path):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                module_name = os.path.relpath(os.path.join(root, file), os.path.dirname(project_path)) \
                                  .replace("/", ".").replace("\\", ".")[:-3]
                try:
                    ipython.magic(f"aimport {module_name}")
                    print(f"Successfully 'aimported' {module_name}")
                except Exception as e:
                    print(f"Failed to aimport {module_name}: {e}")
