# LoL Maya 2023
An attempt to update RiotFileTranslator to Maya 2023.

![](https://i.imgur.com/cRpMpYt.gif)


### Infos:
1. Misc:
    - Add fix for read/write file with suffix in name.
    - All namespaces will be removed in export data. 
    - Use exported file as new scene to improve performance.
2. SKN: 
    - SKN+SKL data in Maya scene: A single mesh that has material and uv assigned on all faces, bound with joints as a skin cluster and have weighted on all vertices.
    - Read: 
        - `33 22 11 00`: V0, V1, V2, V4
        - To read both SKN+SKL as skin cluster, change SKN import options to:
        
            ![](https://i.imgur.com/UiNIMul.png)

        - Add fix for duplicate joint-material name when read both SKN+SKL: 
            - Material that has duplicated name with another joint will be renamed to lowercase. 
            - Example: If there is a joint `Fish` in SKL data, material `Fish` in SKN data will be rename to `fish` in scene.
    - Write: 
        - To export: select the mesh -> use export selection.
        - `33 22 11 00`: V1
        - Add check for limit vertices: 65536 
        - When run into the error: vertices have 4+ influences, those vertices will be selected in scene.
        - When run into the error: vertices have no uv assigned, those vertices will be selected in scene.
        - When run into the error: vertices have no material assigned, those vertices will be selected in scene.
3. SKL:
    - Read: 
        - `r3d2sklt`: V1, V2
        - `C3 4F FD 22`: V0
    - Write:
        - To export: will export with SKN.
        - `C3 4F FD 22`: V0 
        - New SKL data, no need to update/convert.
        - Add check for limit joints: 256
        - Add fix for SKL bad joints order that caused bad animation layers/animation blending, example: Samira reload, Jhin reload,...
            - Extract the original SKL in wad file, rename it to `riot.skl` and put it in export location.
            - While exporting, plugin will attempt to sort your joints order to match `riot.skl` joints order. (not affect the scene)
            - You must have all `riot.skl` joints on your skin cluster.
            - You can add extra joints to your skin, but not allow to remove joints.
            - If there is no `riot.skl` found in the export location, the plugin will export normal way - unsorted.
4. ANM:
    - ANM data in Maya scene: 
        - Translate + Rotate + Scale keyframes of all joints, from time 1 to end time on Time Slider.
        - FPS support: 30/60.
    - Read: 
        - `r3d2canm`
        - `r3d2anmd`: V3, V4, V5
        - To ensure importing FPS from ANM file, change ANM import options to:
            
            ![](https://i.imgur.com/2hJvlGt.png)
    - Write:
        - To export: use export all.
        - `r3d2anmd`: V4 
            - Uncompressed, scaling support.
5. Static object:
    - Static object in Maya scene: 
        - A single triangulated mesh that has UV assigned on all faces.
        - Central point: the translation of mesh's transform oject.
        - Pivot point - SCO only, optional: an additional pivot joint bound with mesh.
            
            ![](https://i.imgur.com/XZFvV3V.png)
    - Read:
        - SCO 
        - SCB: `r3d2Mesh`: V1, V2, V3
    - Write:
        - To export: select the mesh -> use export selection.
        - SCO
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
