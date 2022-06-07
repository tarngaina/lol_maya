# LoL Maya 2023
An attempt to update RiotFileTranslator to Maya 2023.

### Infos:
1. Misc:
    - Add fix for read/write file with suffix in name.
2. SKN: 
    - Read: 
        - `33 22 11 00`: V0, V1, V2, V3, V4
        - Add option to read SKN+SKL as skin cluster.
    - Write: 
        - `33 22 11 00`: V1
        - Add fix for Maya duplicate name system on joint-material nodes.
3. SKL:
    - Read: 
        - `r3d2sklt`: V1, V2
        - `C3 4F FD 22`: V0
    - Write:
        - `C3 4F FD 22`: V0
4. ANM:
    - Read: 
        - `r3d2canm`
        - `r3d2anmd`: V3, V4, V5
        - Must change scene to 30fps before import ANM.
    - Write:
        - `r3d2anmd`: V4
5. Static object:
    - Read:
        - SCO 
        - SCB: `r3d2Mesh`: V1, V2, V3
        - UV not working right now: todo
    - Write: todo



### Installation:

1. Download `plug-ins` and `scripts` folder.

2. Move both folders in `Documents` \ `maya` \ `2023`.

3. In Maya toolbar, select `Windows` > `Settings/Preferences` > `Plug-in Manager`.

4. Tick `Load` / `Auto Load` on the plug-in `lol_maya.py`.
