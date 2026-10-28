@echo off
REM Compile octunillanos.onnx -> octunillanos.axmodel for LLM630 Compute Kit (AX630C = AX620E target)
REM Prereqs: Docker Desktop RUNNING, and the Pulsar2 image loaded once:
REM   docker load -i ax_pulsar2_<version>.tar.gz   (download from https://huggingface.co/AXERA-TECH/Pulsar2)
REM Usage: compile_axmodel.bat <pulsar2_version_tag>     e.g. compile_axmodel.bat 4.2
setlocal
if "%~1"=="" (
  echo Usage: %~nx0 ^<pulsar2_version_tag^>
  exit /b 1
)
cd /d "%~dp0"
if not exist dataset\calib.tar (
  echo Missing dataset\calib.tar - run: python ..\make_calib.py --images ^<folder^>
  exit /b 1
)
docker run --rm -it --net host -v "%cd%":/data pulsar2:%~1 ^
  pulsar2 build --target_hardware AX620E ^
  --input /data/model/octunillanos.onnx ^
  --output_dir /data/output ^
  --config /data/config/octunillanos_config.json ^
  --npu_mode NPU1
echo.
echo Result: output\compiled.axmodel  (rename to octunillanos.axmodel)
endlocal
