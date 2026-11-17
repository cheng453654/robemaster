cmake_minimum_required(VERSION 3.8)
project(rm_robot_description)

find_package(ament_cmake REQUIRED)

# urdf + meshes 是 MoveIt / robot_state_publisher 必需的；
# rviz/config/worlds/models 只在目录存在时安装：仓库版本不含仿真资源
# （worlds/models）和 Gazebo 控制器配置（config），本地仿真工作区仍有
# 这些目录，安装行为不变。
install(
  DIRECTORY urdf meshes
  DESTINATION share/${PROJECT_NAME}
)
foreach(dir rviz config worlds models)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${dir}")
    install(
      DIRECTORY ${dir}
      DESTINATION share/${PROJECT_NAME}
    )
  endif()
endforeach()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
