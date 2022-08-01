# LoL Maya 2023
An attempt to update RiotFileTranslator to Maya 2023.

![](https://i.imgur.com/cRpMpYt.gif)


### File translators:
1. Misc:
    - Add fix for read/write file with suffix in name.
2. SKN: 
    - SKN data in Maya scene: A single mesh that has material and uv assigned on all faces, bound with joints as a skin cluster and have weighted on all vertices.
    - Read: 
        - `33 22 11 00`: V0, V1, V2, V4
        - To read both SKN+SKL as skin cluster, change SKN import options to:
        
            ![](https://i.imgur.com/UiNIMul.png)

        - Add fix for duplicate joint-material name when read both SKN+SKL: 
            - Material that has duplicated name with another joint will be renamed to lowercase. 
            - Example: If there is a joint `Fish` in SKL data, material `Fish` in SKN data will be rename to `fish` in scene.
    - Write: 
        - To export: select the bound mesh -> use export selection.
        - `33 22 11 00`: V1
        - Limit vertices: 65536 
        - Show/select on error: 
            - Vertex: 4+ influences vertex, material shared vertex, non UVs assigned vertex (check on first UV set).
            - Face: invalid triangulation face, non material assigned face, non UVs assigned face.
        - Add check for bad history: skinCluster is not connected directly to meshShape.
        - `riot.skn`:
            - Fix incorrect transparency face and much more shader things,...
            - Extract the original SKN in wad file, rename it to `riot.skn` and put it in export location.
            - While exporting, plugin will attempt to sort your materials to match `riot.skn` materials order. (not affect scene)
            - Custom/extra materials will be added at the end of new materials list.
            - If there is no `riot.skn` found in export location, plugin will export normal way - unsorted.
3. SKL:
    - SKL data in Maya scene: all joints (bound/not bound) in current scene.
    - Read: 
        - `r3d2sklt`: V1, V2
        - `C3 4F FD 22`: V0
    - Write:
        - To export: will export with SKN.
        - `C3 4F FD 22`: V0 
        - Limit joints: 256
        - `riot.skl`:
            - Fix bad animation layers/animation blending,... example: Samira reload while walk/run,...
            - Extract the original SKL in wad file, rename it to `riot.skl` and put it in export location.
            - While exporting, plugin will attempt to sort your joints to match `riot.skl` joints order. (not affect scene)
            - Missing joint from `riot.skl` will be automatically in export data. 
            - Custom/extra joints will be added at the end of new joints list.
            - If there is no `riot.skl` found in export location, plugin will export normal way - unsorted.
        - New SKL data, no need to update/convert.
4. ANM:
    - ANM data in Maya scene: 
        - Translate + Rotate + Scale keyframes of all joints in scene from time 1 to end time on Time Slider.
        - Important: Time Slider must have time 0.
        - FPS support: 30/60.
    - Read: 
        - `r3d2canm`
        - `r3d2anmd`: V3, V4, V5
        - To delete channel data before load ANM, change ANM import options to:
        
            ![](https://i.imgur.com/BWnCj2T.png)

        - To ensure importing FPS from ANM file, change ANM import options to:
            
            ![](https://i.imgur.com/2hJvlGt.png)
    - Write:
        - To export: use export all.
        - `r3d2anmd`: V4 
        - Uncompressed, scaling support.
        - No need to convert with lol2dae or edit 1E hex.
5. Static object:
    - Static object in Maya scene: 
        - A single mesh that has UV assigned on all faces. 
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
        - Show/select on error: 
            - Face: invalid triangulation face, non UVs assigned face.
        - No need to convert with Wooxy.


### Shelf buttons
![](https://i.imgur.com/ZYyknZ9.png)
1. Hover mouse on shelf buttons to read tooltip.
2. Explain some buttons:
    - Namespace buttons: Quickly add/remove a temporary namespace on selected objects.
    - Martin UV helper: move selected UVs to specific corner.
    - Update bind pose button: set current pose as bind pose for skin cluster, require: select single joint of the skin cluster.
    - Freeze joints buttons: Freeze/bake selected/all joints rotation.
    - Mirror joint buttons:
        - L<->R: mirror rotation of a selected joint startswith `L_` or `R_` to the opposite joint.
        - A<->B: mirror rotation of first selected joint to second selected joint.
    - Rebind button: Unbind, delete all history, rebind, copy weight back base on vertex id, on selected skin cluster.
    - 4 influences fix: prune and force max 4 influences on selected skin cluster.


### Installation:
1. [Click here and download latest release.](https://github.com/tarngaina/lol_maya/releases)


2. Extract all `plug-ins`, `prefs` and `scripts` folder to `Documents` \ `maya` \ `2023`.

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
