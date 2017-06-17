
cmake_minimum_required (VERSION 2.8)

PROJECT (mnistTest)


# Requires OpenCV v2.4.1 or later  
FIND_PACKAGE( OpenCV REQUIRED )  
IF (${OpenCV_VERSION} VERSION_LESS 2.4.1)  
    MESSAGE(FATAL_ERROR "OpenCV version is not compatible : ${OpenCV_VERSION}. requires atleast OpenCV v2.4.1")  
ENDIF() 


find_package(Caffe)  
include_directories(${Caffe_INCLUDE_DIRS})  
add_definitions(${Caffe_DEFINITIONS})   

add_executable(${PROJECT_NAME} mnistTest.cpp)
add_executable(mnistTest2 mnistTest2.cpp)
include_directories ( /Users/liuhao/devlibs/deeplearning/caffe/install/include
    /usr/local/include
    /usr/local/cuda/include )

target_link_libraries(${PROJECT_NAME} ${Caffe_LIBRARIES}
                        ${OpenCV_LIBS}  )
 
target_link_libraries(mnistTest2 ${Caffe_LIBRARIES}
                        ${OpenCV_LIBS}  )