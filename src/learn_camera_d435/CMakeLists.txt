cmake_minimum_required(VERSION 3.8)
project(learn_camera_d435)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# 安装 Python 脚本，这样 source install/setup.bash 后可以用
#   ros2 run learn_camera_d435 learn_d435.py
# 或直接运行 learn_d435.py
# PROGRAMS 会保留可执行权限，FILES 不会。
install(PROGRAMS
  src/learn_d435.py
  src/learn_foundationpose.py
  src/d435_camera_node.py
  src/yolo_detect_node.py
  src/foundationpose_node.py
  src/image_view_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# fp_common.py 是三个节点共享的模块，被上面的脚本 import。
# 它不需要可执行权限，用 FILES 安装到同一目录即可。
install(FILES
  src/fp_common.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
