cmake_minimum_required(VERSION 3.15)
project(RLHookLib)

set(CMAKE_CXX_STANDARD 17)

set(PYBIND11_FINDPYTHON ON)

find_package(Git QUIET)
if(GIT_FOUND AND (NOT EXISTS "${PROJECT_SOURCE_DIR}/libs/pybind11/CMakeLists.txt" OR NOT EXISTS "${PROJECT_SOURCE_DIR}/libs/Detours/Detours/README.md"))
    message(STATUS "Cloning pybind11 and detours")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive)
endif()

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/libs/pybind11/CMakeLists.txt" OR NOT EXISTS "${PROJECT_SOURCE_DIR}/libs/Detours/Detours/README.md")
    message(FATAL_ERROR "The submodules pybind11 and/or Detours have not been cloned! Clone them with \"git submodule update --init --recursive\".")
endif()

if(DEFINED ENV{PYTHON_EXEC_PATH})
    set(Python3_EXECUTABLE $ENV{PYTHON_EXEC_PATH} CACHE FILEPATH "Path to Python executable")
else()
    find_program(AUTO_PYTHON_EXEC NAMES python3 python HINTS ENV PATH)
    
    if(AUTO_PYTHON_EXEC)
        set(Python3_EXECUTABLE ${AUTO_PYTHON_EXEC} CACHE FILEPATH "Path to Python executable")
        message(WARNING "Using automatically detected Python executable: ${AUTO_PYTHON_EXEC}")
    else()
        message(FATAL_ERROR "
        Python executable not found. Please set the PYTHON_EXEC_PATH environment variable.

        Examples to set PYTHON_EXEC_PATH:
        
        - For CMD (replace <USERNAME>):
            set PYTHON_EXEC_PATH=C:/Users/<USERNAME>/AppData/Local/Programs/Python/Python314/python.exe

        - For PowerShell (replace <USERNAME>):
            $env:PYTHON_EXEC_PATH = 'C:/Users/<USERNAME>/AppData/Local/Programs/Python/Python314/python.exe'

        After setting the variable, run CMake again.")
    endif()
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

add_subdirectory(libs)
find_package(OpenGL REQUIRED)

add_subdirectory(RLHookDLL)
if(NOT DEFINED ONLY_ADDITIONAL_BINARIES)
    add_subdirectory(PyRLHook)
endif()
add_subdirectory(FunctionAddressGetter)
add_subdirectory(Utils)
