Notes on running on Raspberry Pi 5
==================================

Official Raspberry Pi OS Bookworm is used.

To use vulkan, package "libvulkan-dev" needs to be installed.
Qt tries to load the symbolic link libvulkan.so which is only installed by "libvulkan-dev".

with default api (opengl), the following line shows and no window shows:
    qt.qpa.wayland: eglSwapBuffers failed with 0x300d, surface: 0x0

with "--api vulkan", something gets rendered but the following line (amongst others) gets printed on termination:
    warning: queue "mesa vk display queue" 0x319f4270 destroyed while proxies still attached

with QT_QPA_PLATFORM=xcb, both opengl and vulkan execute with no errors.

to run compute shaders, we need to set QSurfaceFormat to OpenGLES.

