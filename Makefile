BUILD_DIR   := build
BUILD_TYPE  := Debug
APP         := $(BUILD_DIR)/app/triangle
NPROC       := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu)

ifeq ($(shell uname),Darwin)
  HOMEBREW_PREFIX := $(shell brew --prefix)
  RUN_ENV := VK_LAYER_PATH=$(HOMEBREW_PREFIX)/share/vulkan/explicit_layer.d \
             VK_ICD_FILENAMES=$(HOMEBREW_PREFIX)/etc/vulkan/icd.d/MoltenVK_icd.json
endif

.PHONY: build run clean rebuild

build:
	cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
	cmake --build $(BUILD_DIR) -j$(NPROC)

run: build
	$(RUN_ENV) $(APP)

clean:
	rm -rf $(BUILD_DIR)

rebuild: clean build
