@echo off
REM 删除旧的打包文件
IF EXIST dist (
    rmdir /s /q dist
)

REM 打包项目
python setup.py sdist bdist_wheel

REM 上传到 PyPI
python -m twine upload --verbose dist\*

pause