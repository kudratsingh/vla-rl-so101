# Source before running anything on the Mac:  source scripts/mac_env.sh
# MoltenVK gives SAPIEN a Vulkan driver (brew install molten-vk);
# torchcodec 0.2.1 needs FFmpeg 4-7 (brew install ffmpeg@7, keg-only).
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json}"
export DYLD_FALLBACK_LIBRARY_PATH="${DYLD_FALLBACK_LIBRARY_PATH:-/opt/homebrew/opt/ffmpeg@7/lib}"
export MVK_CONFIG_LOG_LEVEL="${MVK_CONFIG_LOG_LEVEL:-0}"
