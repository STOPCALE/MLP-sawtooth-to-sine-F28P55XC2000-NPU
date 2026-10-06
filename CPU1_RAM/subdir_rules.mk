################################################################################
# Automatically-generated file. Do not edit!
################################################################################

SHELL = cmd.exe

# Each subdirectory must supply rules for building sources it contributes
%.obj: ../%.c $(GEN_OPTS) | $(GEN_FILES) $(GEN_MISC_FILES)
	@echo 'C2000 Compiler - building file: "$<"'
	"E:/ccs/ccs/tools/compiler/ti-cgt-c2000_25.11.1.LTS/bin/cl2000" -v28 -ml -mt --cla_support=cla2 --float_support=fpu32 --tmu_support=tmu1 --vcu_support=vcrc -Ooff --include_path="E:/desk/CCS/uart_and_npu" --include_path="E:/ccs/C2000Ware_26_01_00_00" --include_path="E:/desk/CCS/uart_and_npu/device" --include_path="E:/ccs/C2000Ware_26_01_00_00/driverlib/f28p55x/driverlib/" --include_path="E:/ccs/ccs/tools/compiler/ti-cgt-c2000_25.11.1.LTS/include" --define=DEBUG --define=RAM --define=_LAUNCHXL_F28P55X --diag_suppress=10063 --diag_warning=225 --diag_wrap=off --display_error_number --gen_func_subsections=on --abi=eabi --preproc_with_compile --preproc_dependency="$(basename $(<F)).d_raw" --include_path="E:/desk/CCS/uart_and_npu/CPU1_RAM/syscfg" $(GEN_OPTS__FLAG) "$<"
	@echo 'Finished building: "$<"'
	@echo ' '

build-1985648971: ../untitled.syscfg
	@echo 'SysConfig - building file: "$<"'
	"E:/ccs/ccs/utils/sysconfig_1.28.1/sysconfig_cli.bat" -s "E:/ccs/C2000Ware_26_01_00_00/.metadata/sdk.json" -d "F28P55x" -p "128PDT" -r "F28P55x_128PDT" --script "E:/desk/CCS/uart_and_npu/untitled.syscfg" -o "syscfg" --compiler ccs
	@echo 'Finished building: "$<"'
	@echo ' '

syscfg/board.c: build-1985648971 ../untitled.syscfg
syscfg/board.h: build-1985648971
syscfg/board.cmd.genlibs: build-1985648971
syscfg/board.opt: build-1985648971
syscfg/board.json: build-1985648971
syscfg/pinmux.csv: build-1985648971
syscfg/c2000ware_libraries.cmd.genlibs: build-1985648971
syscfg/c2000ware_libraries.opt: build-1985648971
syscfg/c2000ware_libraries.c: build-1985648971
syscfg/c2000ware_libraries.h: build-1985648971
syscfg/clocktree.h: build-1985648971
syscfg: build-1985648971

syscfg/%.obj: ./syscfg/%.c $(GEN_OPTS) | $(GEN_FILES) $(GEN_MISC_FILES)
	@echo 'C2000 Compiler - building file: "$<"'
	"E:/ccs/ccs/tools/compiler/ti-cgt-c2000_25.11.1.LTS/bin/cl2000" -v28 -ml -mt --cla_support=cla2 --float_support=fpu32 --tmu_support=tmu1 --vcu_support=vcrc -Ooff --include_path="E:/desk/CCS/uart_and_npu" --include_path="E:/ccs/C2000Ware_26_01_00_00" --include_path="E:/desk/CCS/uart_and_npu/device" --include_path="E:/ccs/C2000Ware_26_01_00_00/driverlib/f28p55x/driverlib/" --include_path="E:/ccs/ccs/tools/compiler/ti-cgt-c2000_25.11.1.LTS/include" --define=DEBUG --define=RAM --define=_LAUNCHXL_F28P55X --diag_suppress=10063 --diag_warning=225 --diag_wrap=off --display_error_number --gen_func_subsections=on --abi=eabi --preproc_with_compile --preproc_dependency="syscfg/$(basename $(<F)).d_raw" --include_path="E:/desk/CCS/uart_and_npu/CPU1_RAM/syscfg" --obj_directory="syscfg" $(GEN_OPTS__FLAG) "$<"
	@echo 'Finished building: "$<"'
	@echo ' '


