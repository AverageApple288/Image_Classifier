cmake_minimum_required(VERSION 4.1)
project(Image_Classifier)

set(CMAKE_CXX_STANDARD 26)

find_package(PkgConfig REQUIRED)
find_package(OpenMP REQUIRED)
pkg_check_modules(GTKMM REQUIRED gtkmm-4.0)

add_executable(Image_Classifier main.cpp
        src/gui/Window.cpp
        include/gui/Window.h
        src/utils/Utilities.cpp
        include/utils/Utilities.h
        src/utils/Matrix3D.cpp
        include/utils/Matrix3D.h
        src/utils/DataLoader.cpp
        include/utils/DataLoader.h
        src/layers/Layer.cpp
        include/layers/Layer.h
        src/layers/Conv2DLayer.cpp
        include/layers/Conv2DLayer.h
        src/layers/PoolingLayer.cpp
        include/layers/PoolingLayer.h
        src/layers/DenseLayer.cpp
        include/layers/DenseLayer.h
        src/model/NeuralNetwork.cpp
        include/model/NeuralNetwork.h
        src/model/LossFunction.cpp
        include/model/LossFunction.h
        src/model/Optimiser.cpp
        include/model/Optimiser.h
        src/model/Model.cpp
        include/model/Model.h
)

target_include_directories(Image_Classifier PRIVATE ${GTKMM_INCLUDE_DIRS})
target_link_libraries(
        Image_Classifier PRIVATE ${GTKMM_LIBRARIES}
        zip
        OpenMP::OpenMP_CXX
)
target_compile_options(Image_Classifier PRIVATE ${GTKMM_CFLAGS_OTHER})