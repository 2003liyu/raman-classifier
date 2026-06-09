from PyInstaller.__main__ import run
import platform
import shutil
import os
from datetime import datetime
import subprocess


version = "0.0.0.1"
company_name = "河南农业大学"
company_en_name = "henau"
app_name = "拉曼光谱分类器"
app_en_name = "raman_classifier"


def install(source, destination):
    if not os.path.exists(source):
        return
    
    destination = os.path.abspath(destination).replace("\\", "/")

    if os.path.isfile(source):
        if os.path.isdir(destination):
            destination = os.path.join(destination, os.path.basename(source))

        destination_dir = os.path.dirname(destination)
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        if os.path.exists(destination):
            source_mtime = os.path.getmtime(source)
            destination_mtime = os.path.getmtime(destination)
            if destination_mtime >= source_mtime:
                print(f"-- Up-to-date: {destination}")
            else:
                shutil.copy2(source, destination)
                print(f"-- Installing: {destination}")
        else:
            shutil.copy2(source, destination)
            print(f"-- Installing: {destination}")
    elif os.path.isdir(source):
        if not os.path.exists(destination):
            os.makedirs(destination)

        for root, _, files in os.walk(source):
            for file in files:
                source_file = os.path.join(root, file)
                relative_path = os.path.relpath(source_file, source)
                target_file = os.path.join(destination, relative_path).replace("\\", "/")
                target_folder = os.path.dirname(target_file)
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)

                if os.path.exists(target_file):
                    source_mtime = os.path.getmtime(source_file)
                    target_mtime = os.path.getmtime(target_file)
                    if target_mtime >= source_mtime:
                        print(f"-- Up-to-date: {target_file}")
                    else:
                        shutil.copy2(source_file, target_file)
                        print(f"-- Installing: {target_file}")
                else:
                    shutil.copy2(source_file, target_file)
                    print(f"-- Installing: {target_file}")

versions = version.split(".")
self_folder:str = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
domain_str:str = f"com.{company_en_name}.{app_en_name.replace('_', '').replace('-', '')}"
distpath:str = f"{self_folder}/installer/packages/{domain_str}"

version_str = f"""
VSVersionInfo(
  ffi=FixedFileInfo(
    # filevers and prodvers should be always a tuple with four items: ({versions[0]}, {versions[1]}, {versions[2]}, {versions[3]})
    # Set not needed items to zero 0.
    filevers=({versions[0]}, {versions[1]}, {versions[2]}, {versions[3]}),
    prodvers=({versions[0]}, {versions[1]}, {versions[2]}, {versions[3]}),
    # Contains a bitmask that specifies the valid bits 'flags'r
    mask=0x3f,
    # Contains a bitmask that specifies the Boolean attributes of the file.
    flags=0x0,
    # The operating system for which this file was designed.
    # 0x4 - NT and there is no need to change it.
    OS=0x4,
    # The general type of file.
    # 0x1 - the file is an application.
    fileType=0x1,
    # The function of the file.
    # 0x0 - the function is not defined for this fileType
    subtype=0x0,
    # Creation date and time stamp.
    date=(0, 0)
    ),
  kids=[
    StringFileInfo(
      [
      StringTable(
        u'000004b0',
        [StringStruct(u'Comments', u''),
        StringStruct(u'CompanyName', u'{company_name}'),
        StringStruct(u'FileDescription', u'{app_name}'),
        StringStruct(u'FileVersion', u'{version}'),
        StringStruct(u'InternalName', u'raman_classifier'),
        StringStruct(u'LegalCopyright', u'© {datetime.now().year} {company_name}'),
        StringStruct(u'LegalTrademarks', u''),
        StringStruct(u'OriginalFilename', u'raman_classifier.exe'),
        StringStruct(u'ProductName', u'{app_name}'),
        StringStruct(u'ProductVersion', u'{version}'),
        StringStruct(u'Assembly Version', u'{version}')])
      ]), 
    VarFileInfo([VarStruct(u'Translation', [0, 1200])])
  ]
)
"""

config_xml = f"""<Installer>
    <Name>{app_name}</Name>
    <Version>{version}</Version>
    <Title>{app_name}</Title>
    <Publisher>{company_name}</Publisher>
    <StartMenuDir>{app_en_name}</StartMenuDir>
</Installer>
"""

package_xml = f"""<Package>
    <DisplayName>{app_name}</DisplayName>
    <Description>{app_name}</Description>
    <Version>{version}</Version>
    <ReleaseDate>{datetime.now().strftime('%Y-%m-%d')}</ReleaseDate>
    <Default>true</Default>
</Package>
"""

with open("version.py", "w", encoding="utf-8") as out_file:
    out_file.write(version_str)

if os.path.isdir("build"):
    shutil.rmtree("build")

if os.path.isdir(f"{app_en_name}.egg-info"):
    shutil.rmtree(f"{app_en_name}.egg-info")

if os.path.isdir(distpath):
    shutil.rmtree(distpath)

icon_path = f"{app_en_name}/icons/{app_en_name}."
if platform.system() == "Windows":
    icon_path += "ico"
else:
    icon_path += "png"

cmds = [
    "-D",
    "main.py",
    "-n", f"{app_en_name}",
    "--icon", icon_path,
    "--version-file", "version.py",
    "--add-data", f"{app_en_name}/icons/{app_en_name}.png;{app_en_name}/icons",
    "--distpath", f"{self_folder}/installer/packages/{domain_str}",
    # "-w", # 注释掉即可打开调试窗口
]
run(cmds)
os.rename(f"{self_folder}/installer/packages/{domain_str}/{app_en_name}", f"{self_folder}/installer/packages/{domain_str}/data")

if not os.path.isdir(f"{self_folder}/installer/config"):
    os.makedirs(f"{self_folder}/installer/config")

with open(f"{self_folder}/installer/config/config.xml", "w", encoding="utf-8") as config_out_file:
    config_out_file.write(config_xml)

if not os.path.isdir(f"{self_folder}/installer/packages/{domain_str}/meta"):
    os.makedirs(f"{self_folder}/installer/packages/{domain_str}/meta")

if not os.path.isdir(f"{self_folder}/installer/packages/{domain_str}/data"):
    os.makedirs(f"{self_folder}/installer/packages/{domain_str}/data")

with open(f"{self_folder}/installer/packages/{domain_str}/meta/package.xml", "w", encoding="utf-8") as package_out_file:
    package_out_file.write(package_xml)

subprocess.check_call([
    "binarycreator", "-c", f"{self_folder}/installer/config/config.xml",
    "-p", f"{self_folder}/installer/packages",
    f"{self_folder}/installer/{app_en_name.replace('_', '-')}-x64-installer-v{version}.exe"
])