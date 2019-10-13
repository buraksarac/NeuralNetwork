################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/ApplicationParameters.cpp \
../src/Fmincg.cpp \
../src/GradientParameter.cpp \
../src/IOUtils.cpp \
../src/NeuralNetwork.cpp \
../src/NeuralProcessor.cpp 

OBJS += \
./src/ApplicationParameters.o \
./src/Fmincg.o \
./src/GradientParameter.o \
./src/IOUtils.o \
./src/NeuralNetwork.o \
./src/NeuralProcessor.o 

CPP_DEPS += \
./src/ApplicationParameters.d \
./src/Fmincg.d \
./src/GradientParameter.d \
./src/IOUtils.d \
./src/NeuralNetwork.d \
./src/NeuralProcessor.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


