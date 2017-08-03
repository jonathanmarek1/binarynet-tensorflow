ImageNet example using camera on Android. Tested on Nexus 5.

Install required tools (debian sid):
```
# clang
apt install clang-5.0 lld-5.0
# java
apt install openjdk-8-jdk
# android tools
apt install dalvik-exchange zipalign aapt libandroid-23-java libandroid-tools-sdklib-java
```

You will also need NDK headers and libraries.


You can either generate your own weights and code using the scripts or use these:

[XNORNET_BWN_OUTPUT](https://github.com/jonathanmarek1/binarynet-tensorflow/releases/download/test/XNORNET_BWN_OUTPUT.zip)


Build the apk:
```
ARCH="armeabi-v7a" CFLAGS="-target armv7a-none-linux-android -mcpu=krait -mfpu=neon-vfpv4 -DNUM_THREAD=2" NDK="/media/test/app/android-ndk-r15b" sh make
```
