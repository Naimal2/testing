Follow these steps:
1- clone repository into new directory
2- run `pipenv install -r requirements.txt`
3- run `pipenv shell` and you will be into environment
4- run app
5- if app is not runned then use `pip install -r requirements.txt`
6- make sure all libraries are installed 
7- run the app

if this error occures:
<!-- qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: eglfs, linuxfb, minimal, minimalegl, offscreen, vnc, wayland-egl, wayland, wayland-xcomposite-egl, wayland-xcomposite-glx, webgl, xcb.

Aborted (core dumped) -->

1- Check the Qt platform plugins directory: Verify that the necessary Qt platform plugins are present in the application's directory. Look for a folder named "platforms" that should contain the required plugins. The plugins you mentioned in the error message are typically located in the following directory: <Qt Installation Directory>\plugins\platforms\. Make sure the xcb.dll file is present in this folder.

2- Verify the PATH environment variable: Ensure that the Qt installation directory is added to the PATH environment variable. Follow these steps:

    Press Win + R to open the Run dialog.
    Type "sysdm.cpl" and press Enter to open the System Properties window.
    In the System Properties window, click on the "Advanced" tab.
    Click on the "Environment Variables" button.
    In the "System variables" section, locate the "Path" variable and click "Edit".
    Add the path to the Qt installation directory (e.g., C:\Qt\Qt_version\bin) to the list of paths. If the "Path" variable doesn't exist, click "New" to create it.
    Click "OK" to save the changes.

3- Reinstall or repair the Qt installation: If the above steps don't resolve the issue, try reinstalling or repairing your Qt installation. Download the latest version of Qt for Windows from the official website (https://www.qt.io/) and run the installer. Choose the option to repair or reinstall the installation.

4- Try a different platform plugin: Similar to the Linux solution, you can force the use of a different platform plugin on Windows. Instead of using "xcb," you can try using "windows" or "minimal" as the platform plugin. Modify your command as follows:

/path/to/python/python.exe -platform windows /path/to/your/script.py
Replace "/path/to/python/python.exe" with the path to your Python interpreter, and "/path/to/your/script.py" with the actual path to your script.

5- By following these steps, you should be able to resolve the Qt platform plugin error on Windows.





