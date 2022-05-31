# LoL Maya 2023
An attempt to update RiotFileTranslator to Maya 2023

Infos:
1. Misc:
    - Add fix for read/write file with suffix in name.
2. SKN: 
    - Read old SKN, new SKN.
    - Write out legacy SKN.
    - Add fix for exporting SKN that has same node name of joint-material.
    - Add option to read both SKN+SKL as skin cluster with weights (same filename same location).
3. SKL:
    - Read old SKL, new SKL.
    - Write out new SKL.
4. ANM: todo
5. SCO/SCB: todo



Installation:

1. Download `plug-ins` and `scripts` folder.

2. Move both folders in `Documents\maya\2023`.

3. In Maya toolbar, select `Windows` > `Settings/Preferences` > `Plug-in Manager`.

4. Tick `Load` / `Auto Load` on the plug-in `lol_maya.py`.