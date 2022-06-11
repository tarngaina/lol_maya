# LoL Maya 2023
An attempt to update RiotFileTranslator to Maya 2023.

![](https://i.imgur.com/cRpMpYt.gif)


### Infos:
1. Misc:
    - Add fix for read/write file with suffix in name.
    - Add fix for SKL bad joints order which caused bad animation blending like: Samira reload, Jhin reload,...
        - Extract the original skl in wad file, rename it to `riot.skl`, put it in export location.
        - When exporting, plugin will attempt to sort your joints to match `riot.skl` joints order. (not affect the scene)
        - You must have all `riot.skl` joints on your skin cluster.
        - You can add extra joints to your skin, but not allow to remove joints.
        - If there is no `riot.skl` found in the export location, the plugin will export the normal way.
2. SKN: 
    - Read: 
        - `33 22 11 00`: V0, V1, V2, V3, V4
        - To read both SKN/SKL as skin cluster, change SKN import options to:
        
            ![](https://i.imgur.com/UiNIMul.png)
    - Write: 
        - `33 22 11 00`: V1
        - Add fix for Maya duplicate name system on joint-material nodes, example: joint `Fish` and material `Fish1` will export as `Fish`. (no effect on index 2, 3, 4,...)
3. SKL:
    - Read: 
        - `r3d2sklt`: V1, V2
        - `C3 4F FD 22`: V0
    - Write:
        - `C3 4F FD 22`: V0 (new SKL, no need to update/convert after)
4. ANM:
    - Read: 
        - `r3d2canm`
        - `r3d2anmd`: V3, V4, V5
        - To ensure 30FPS import, change ANM import options to:
        
            ![](https://i.imgur.com/2hJvlGt.png)
    - Write:
        - `r3d2anmd`: V4 (uncompressed, scaling support)
5. Static object:
    - Read:
        - SCO 
        - SCB: `r3d2Mesh`: V1, V2, V3
    - Write:
        - SCO: 
            - Pivot point (optional): is translation of a joint that bound with the mesh.

                ![](https://i.imgur.com/XZFvV3V.png)
        - SCB: `r3d2Mesh`: V3 (no need to convert with Wooxy)



### Installation:
1. [Click here and download latest release.](https://github.com/tarngaina/lol_maya/releases)


2. Extract both `plug-ins` & `scripts` to `Documents` \ `maya` \ `2023`.

    ![](https://i.imgur.com/OuXcoD7.png)

3. In Maya toolbar, select `Windows` > `Settings/Preferences` > `Plug-in Manager`.

    ![](https://i.imgur.com/fawHenl.png)

4. Tick `Load` / `Auto Load` on the plug-in `lol_maya.py`.

    ![](https://i.imgur.com/D0Za7BU.png)



### External Links:

- [LeagueFileTranlastor](https://github.com/LoL-Fantome/LeagueFileTranslator)

- [LeagueToolKit](https://github.com/LoL-Fantome/LeagueToolkit)

- [LoL2Blender](https://github.com/WorldSEnder/LoL2Blender)

- [Maya SDK Reference](https://help.autodesk.com/cloudhelp/2023/ENU/Maya-SDK/cpp_ref/modules.html)
