cmake_minimum_required(VERSION 3.10)
set(CMAKE_BUILD_TYPE Release)
project(pointcloud_aligner VERSION 1.0.0 LANGUAGES CXX C)

# Output directories - always use the same build/ directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")


# Use ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

# Find required packages
find_package(OpenCV REQUIRED)
find_package(PCL 1.15 REQUIRED)
find_package(yaml-cpp REQUIRED)

# Find Qt
find_package(Qt5 REQUIRED COMPONENTS Core Widgets)

# Find VTK with specific components and disable MPI
set(VTK_REQUIRED_COMPONENTS
    CommonCore
    CommonDataModel
    CommonExecutionModel
    RenderingCore
    RenderingOpenGL2
    InteractionWidgets
    FiltersCore
    IOCore
    IOXML
    IOImage
    GUISupportQt
)

find_package(VTK REQUIRED COMPONENTS ${VTK_REQUIRED_COMPONENTS})

# Set up include directories
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${YAMLCPP_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${VTK_INCLUDE_DIRS})

# Set up library directories and definitions
link_directories(${PCL_LIBRARY_DIRS})
link_directories(${VTK_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


# Add the color_icp library as static
add_library(color_icp STATIC src/color_icp.cpp)
target_link_libraries(color_icp 
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    ${VTK_LIBRARIES}
)
target_compile_definitions(color_icp PRIVATE PCL_SILENCE_MALLOC_WARNING=1)


# Check if Qt5 is available for the GUI version
find_package(Qt5 QUIET COMPONENTS Core Widgets OpenGL)
if(Qt5_FOUND)
    # Also need VTK Qt integration
    find_package(VTK QUIET COMPONENTS GUISupportQt)
    
    if(VTK_GUISupportQt_FOUND)
        # QT MANUAL ALIGNER - Qt-based GUI version with 12 buttons for 6 DOF control
        add_executable(pointcloud_aligner apps/qt_manual_alignment_tool.cpp)

        # Enable Qt5 automatic MOC processing
        set_target_properties(pointcloud_aligner PROPERTIES
            AUTOMOC ON
            AUTORCC ON
            AUTOUIC ON
        )
        
        target_link_libraries(pointcloud_aligner
            ${OpenCV_LIBS}
            ${PCL_LIBRARIES}
            ${VTK_LIBRARIES}
            Qt5::Core
            Qt5::Widgets
            Qt5::OpenGL
            VTK::GUISupportQt
            color_icp
        )
        
        target_compile_definitions(pointcloud_aligner PRIVATE PCL_SILENCE_MALLOC_WARNING=1)
        
        message(STATUS "Qt5 and VTK-Qt found - building pointcloud_aligner")
    else()
        message(STATUS "VTK-Qt integration not found - skipping pointcloud_aligner")
    endif()
else()
    message(STATUS "Qt5 not found - skipping pointcloud_aligner")
endif()
