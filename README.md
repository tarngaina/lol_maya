# LoL Maya 2023
An attempt to update RiotFileTranslator to Maya 2023.

![](https://i.imgur.com/cRpMpYt.gif)


### Infos:
1. Misc:
    - Add fix for read/write file with suffix in name.
    - SKN+SKL data in Maya scene: A single mesh that bound with joints as a skin cluster, have weighted on all vertices, and has materials assigned on all faces. -> use export selection
    - ANM data in Maya scene: Translate+rotate+scale keyframes from time 1 to end time of all joints on Time Slider. -> use export all
    - SCO/SCB data in Maya scene: A single mesh has 1 material assigned (;bound with a single joint to write pivot point - SCO only/optional). -> use export selection
2. SKN: 
    - Read: 
        - `33 22 11 00`: V0, V1, V2, V4
        - To read both SKN+SKL as skin cluster, change SKN import options to:
        
            ![](https://i.imgur.com/UiNIMul.png)

        - Add fix for duplicate joint-material name when read both SKN+SKL, all duplicated material name will be lowercase. Example: If there is a joint named `Fish` in SKL data, material `Fish` in SKN data will be rename to `fish` in maya scene.
    - Write: 
        - `33 22 11 00`: V1
        - Add check for limit vertices: 65536 
        - When run into the error: vertices have 4+ influences, those vertices will be selected in scene.
        - When run into the error: vertices have no uv assigned, those vertices will be selected in scene.
3. SKL:
    - Read: 
        - `r3d2sklt`: V1, V2
        - `C3 4F FD 22`: V0
    - Write:
        - `C3 4F FD 22`: V0 
        - New SKL data, no need to update/convert.
        - Add check for limit joints: 256
        - Add fix for SKL bad joints order that caused bad animation layers/animation blending like: Samira reload, Jhin reload,...
            - Extract the original SKL in wad file, rename it to `riot.skl`, put it in export location.
            - When exporting, plugin will attempt to sort your joints to match `riot.skl` joints order. (not affect the scene)
            - You must have all `riot.skl` joints on your skin cluster.
            - You can add extra joints to your skin, but not allow to remove joints.
            - If there is no `riot.skl` found in the export location, the plugin will export normal way - unsorted.
4. ANM:
    - Read: 
        - `r3d2canm`
        - `r3d2anmd`: V3, V4, V5
        - To ensure importing FPS from ANM file, change ANM import options to:
        
            ![](https://i.imgur.com/2hJvlGt.png)
    - Write:
        - `r3d2anmd`: V4 
            - Uncompressed, scaling support.
5. Static object:
    - Read:
        - SCO 
        - SCB: `r3d2Mesh`: V1, V2, V3
    - Write:
        - SCO: 
            - Pivot point (optional): is translation of a joint that bound with the mesh.

                ![](https://i.imgur.com/XZFvV3V.png)
        - SCB: `r3d2Mesh`: V3 
            - No need to convert with Wooxy.



### Installation:
1. [Click here and download latest release.](https://github.com/tarngaina/lol_maya/releases)


2. Extract both `plug-ins` folder & `scripts` folder to `Documents` \ `maya` \ `2023`.

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
