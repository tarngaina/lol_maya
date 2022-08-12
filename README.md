# LoL Maya 2023
An attempt to update RiotFileTranslator to Maya 2023.

![](https://i.imgur.com/cRpMpYt.gif)

### Installation:
1. [Click here and download latest release.](https://github.com/tarngaina/lol_maya/releases)


2. Extract all `plug-ins`, `prefs` and `scripts` folder to `Documents` \ `maya` \ `2023`.

    ![](https://i.imgur.com/OuXcoD7.png)

3. In Maya toolbar, select `Windows` > `Settings/Preferences` > `Plug-in Manager`.

    ![](https://i.imgur.com/fawHenl.png)

4. Tick `Load` / `Auto Load` on the plug-in `lol_maya.py`.

    ![](https://i.imgur.com/D0Za7BU.png)


### File translators:
1. Misc:
    - Add fix for read/write file with suffix in name.
    - Export base on original file:
        - How to: have at least 1 of 2 files below in export location:
            - `riot_{name of export file}.EXT` (take priority first)
            - `riot.EXT`
        - File type support (EXT):
            - SKN: For fixing incorrect transparent faces on champions.
            - SKL: For fixing bad animation blending of champions that have animation layers.
            - SCO, SCB: For fixing incorrect pivot and central point.
        - Example: If you want to export modified `yone_base.skl` base on original file, you must have either `riot_yone_base.skl` or `riot.skl` in export location; if you have both of them, `riot_yone_base.skl` will take priority.
2. SKN: 
    - SKN data in Maya scene: 
        - Combined method: A single mesh that has material and uv assigned on all faces, bound with joints as a skin cluster and have weight painted on all vertices.
        - Separated method: A group of meshes, 1 mesh = material assigned; all meshes have uv assigned on all faces; all meshes bound with joints, 1 mesh = 1 skin cluster; all meshes have weight painted on all vertices.
    - Read: 
        - `33 22 11 00`: V0, V1, V2, V4
        - SKN import options:
            - Import skeleton: load with SKL as skin cluster.
                - Material that has duplicated name with another joint will be renamed to full lowercase letters. 
                - Example: If there is a joint `Fish`, material `Fish` will be rename to `fish` in scene.
            - Import mesh separated by material: load SKN as group of meshes, 1 mesh = 1 material assigned.
        
            ![](https://i.imgur.com/HhoFMLu.png)

    - Write: 
        - To export: 
            - Combined method: select the bound mesh -> use export selection.
            - Separated method: select the group of meshes -> use export selection.
        - `33 22 11 00`: V1
        - Limit vertices: 65536 
        - Show/select component on error: 
            - Vertex: 4+ influences vertex, material shared vertex, non UVs assigned vertex (check on first UV set).
            - Face: invalid triangulation face, non material assigned face, non UVs assigned face.
3. SKL:
    - SKL data in Maya scene: all joints (bound/not bound) in current scene.
    - Read: 
        - `r3d2sklt`: V1, V2
        - `C3 4F FD 22`: V0
    - Write:
        - To export: will be exported with SKN.
        - `C3 4F FD 22`: V0 
        - Limit joints: 256
        - New SKL data, no need to update/convert.
4. ANM:
    - ANM data in Maya scene: 
        - Translate + Rotate + Scale keyframes of all joints in scene from time 1 to end time on Time Slider.
        - **Important**: Time Slider playback must have time 0 even though animation start time is 1.
        - FPS support: 30/60.
    - Read: 
        - `r3d2canm`
        - `r3d2anmd`: V3, V4, V5
        - To delete channel data before load ANM, change ANM import options to:
        
            ![](https://i.imgur.com/BWnCj2T.png)

        - To ensure importing FPS and animation range from ANM file, change ANM import options to:
            
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
        - Show/select component on error: 
            - Face: invalid triangulation face, non UVs assigned face.
        - No need to convert with Wooxy.


### Shelf buttons
![](https://i.imgur.com/jywPEoy.png)
1. Hover mouse on shelf buttons to read tooltip.
2. Explain some buttons:
    - Namespace buttons: Quickly add/remove a temporary namespace on selected objects.
    - Separated mesh button: Separate selected mesh by materials. (Do not work with bound mesh)
    - Martin UV helper: move selected UVs to specific corner.
    - Update bind pose button: set current pose as bind pose for skin cluster, require: select single joint of the skin cluster.
    - Freeze joints buttons: Freeze/bake selected/all joints rotation.
    - Mirror joint buttons:
        - L<->R: mirror rotation of a selected joint startswith `L_` or `R_` to the opposite joint.
        - A<->B: mirror rotation of first selected joint to second selected joint.
    - 4 influences fix: prune and force max 4 influences on selected skin cluster.
    - Rebind button: Quickly unbind, delete all history, rebind selected skin cluster then copy weights back base on vertex ID.
    - Copy group weights: copy weights from first selected group to second selected group base on mesh name inside the group.

### External Links:

- [LeagueFileTranlastor](https://github.com/LoL-Fantome/LeagueFileTranslator)

- [LeagueToolKit](https://github.com/LoL-Fantome/LeagueToolkit)

- [LoL2Blender](https://github.com/WorldSEnder/LoL2Blender)

- [Maya SDK Reference](https://help.autodesk.com/cloudhelp/2023/ENU/Maya-SDK/cpp_ref/modules.html)
