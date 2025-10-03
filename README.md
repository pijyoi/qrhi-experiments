Notes on running on Raspberry Pi 5
==================================

To use vulkan, package "libvulkan-dev" needs to be installed.
Qt tries to load the symbolic link libvulkan.so which is only installed by "libvulkan-dev".

With Raspberry Pi OS Trixie, both opengl and vulkan run with no issues.

The compute shader example needs to be executed with either --opengles or --api vulkan



With Raspberry Pi OS Bookworm, the following issues were encountered.

with default api (opengl), the following line shows and no window shows:
    qt.qpa.wayland: eglSwapBuffers failed with 0x300d, surface: 0x0

with "--api vulkan", something gets rendered but the following line (amongst others) gets printed on termination:
    warning: queue "mesa vk display queue" 0x319f4270 destroyed while proxies still attached

with QT_QPA_PLATFORM=xcb, both opengl and vulkan execute with no errors.

to run compute shaders, we need to set QSurfaceFormat to OpenGLES.

