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
        - Combined method: A single mesh that has materials and UVs assigned (first UV set only) on all faces, face normals point inward, bound with joints as a skin cluster and have weight painted on all vertices.

            ![](https://i.imgur.com/P6S36i0.png)
        - Separated method: A group of meshes, all meshes have materials and UVs assigned (first UV set only) on all faces, face normals point inward, bound with joints as skin clusters and have weight painted on all vertices.

            ![](https://i.imgur.com/m1dfCgR.png)
    - Read: 
        - `33 22 11 00`: V0, V1, V2, V4
        - SKN import options:
            - Import skeleton: load with SKL as skin cluster.
                - Material that has duplicated name with another joint will be renamed to full lowercase letters. 
                - Example: If there is a joint `Fish`, material `Fish` will be rename to `fish` in scene.
            - Import mesh separated by material: load SKN as group of meshes, separated by materials.
        
            ![](https://i.imgur.com/HhoFMLu.png)

    - Write: 
        - To export: 
            - Combined method: select the bound mesh -> use export selection.
            - Separated method: select the group of bound meshes -> use export selection.
        - `33 22 11 00`: V1
        - Limit vertices: 65536 
        - Show/select component on error: 
            - Vertex: 4+ influences vertex, material shared vertex, non UVs assigned vertex.
            - Face: invalid triangulation face, non material assigned face, non UVs assigned face.
3. SKL:
    - SKL data in Maya scene: all joints that either bound or not bound in scene.
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
        - To delete Channels before load ANM, change ANM import options to:
        
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
        - A single mesh that has UVs assigned (first UV set only) on all faces, face normals point inward.
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
![](https://i.imgur.com/NHPDz4D.png)
1. Hover mouse on shelf buttons to read tooltip.
2. Explain some buttons:
    - Namespace buttons: Quickly add/remove a temporary namespace on selected objects.
    - Separated mesh button: Separate selected mesh by materials.
    - Fix shared vertices button.
    - Martin UV helper: move selected UVs to specific corner.
    - Update bind pose button: set current pose as bind pose for skin cluster, require: select single joint of the skin cluster.
    - Freeze joints buttons: Freeze/bake selected/all joints rotation.
    - Mirror joint buttons:
        - L<->R: mirror rotation of a selected joint startswith `L_` or `R_` to the opposite joint.
        - A<->B: mirror rotation of first selected joint to second selected joint.
    - 4 influences fix: prune and force max 4 influences on selected skin cluster.
    - Rebind button: Quickly unbind, delete all history, rebind selected skin cluster.
    - Copy group weights: copy weights from first selected group to second selected group base on mesh name inside the group.

### MAPGEO
1. MAPGEO data in Maya scene:
    - A group of meshes that have materials and UVs assigned (first UV set only) on all faces, face normals point inward.

        ![](https://i.imgur.com/AlmIqQV.png)
    - Materials:
        - Material names in MAPGEO files or in BIN files are used with `/`, this character can't be used in Maya, so all `/` will be converted to `__` when import, and will be converted back when export.
        - Example: 
            - In mapgeo or bin: `Maps/KitPieces/Howling_Abyss/Materials/Keep_inst`
            - In Maya: `Maps__KitPieces__Howling_Abyss__Materials__Keep_inst`
        - Material type that used in Maya for mapgeo by default is Standard Surface, lambert or any other materials can work but not fully support with import and export materials.
    - Layer: 
        - A map will have 8 layers, equal to 8 `set` in Maya.

            ![](https://i.imgur.com/uxlcYsw.png)
        - If an object is assigned to a layer, it will appear on that layer.
        - An object can be assigned to multiple layers, if it is assigned on all 8 layers, it will appear on all 8 layers.
        - Layer in Summoner Rift (SR): (reason for layer to exists)
            - Layer 1: Base
            - Layer 2: Infernal
            - Layer 3: Mountain
            - Layer 4: Ocean
            - Layer 5: Cloud
            - Layer 6: Hextech
            - Layer 7: Chemtech
            - Layer 8: Unknown
            - Example in SR: if mesh assigned to set2 -> that object will appear in layer 2 - Infernal map.
        - Layer in Aram / other map: objects are assigned to all layer.
        
    - Lightmap:
        - A second texture + UVs to store light data in, will blend with main texture inside game.

            ![](https://i.imgur.com/8mlKG0M.png)
        - Can set lightmap through an attribute on each mesh's transform in Attribute Editor, this attribute can be added to mesh's transform by a shelf button.

            ![](https://i.imgur.com/GClAouY.png)
        - Lightmap full path:
            `ASSETS/Maps/Lightmaps/Maps/MapGeometry/{group's name}/Base/{lightmap value}`
            - Example in Aram: `ASSETS/Maps/Lightmaps/Maps/MapGeometry/Map12/Base/1.dds`
        - You don't need to have lightmap if you can bake light into main texture, like Riot did in SR.

2. Read:
    - `OEGM`: V5, V6, V7, V9, V11
3. Write:
    - To export: select the group of meshes -> use export selection.
    - `OEGM`: V11
    - Limit vertices for each mesh: 65536.
    - Mesh's transform that has lightmap attribute value must have a UV set named `lightmap`, to write out as lightmap UV, and the first UV set will be diffuse UV.
    - Freeze all meshes's transform first before export, because mesh's transform values are not supported in mapgeo data.

        ![](https://i.imgur.com/8eZIWQU.png)
4. Shelf buttons:

    ![](https://i.imgur.com/2hlr9Y0.png)

    Explain from left to right: 
    - Toggle all / non layers on selected mesh.
    - Toggle 1 - 8 layers on selected mesh.
    - Rename all material path in scene with input.
    - Fix shared vertices on all meshes in scene.
    - Set all black emission weight to 0.
    - Import material py: read material py file to import textures.
    - Export material json: export all materials + textures in scene to a json file, can read by `Avatar (made by Killery)` to convert back to material py.
        - Lambert/other material:
            - Color, Transparency = Diffuse
        - Standard surface / Arnold's Standard surface:
            - Base = Diffuse 
            - Transmission = Glow
            - Coat = Mask
            - Emission = Emissive
    - Add lightmap attribute to selected mesh.

### External Links:

- [LeagueFileTranlastor](https://github.com/LoL-Fantome/LeagueFileTranslator)

- [LeagueToolKit](https://github.com/LoL-Fantome/LeagueToolkit)

- [LoL2Blender](https://github.com/WorldSEnder/LoL2Blender)

- [Maya SDK Reference](https://help.autodesk.com/cloudhelp/2023/ENU/Maya-SDK/cpp_ref/modules.html)

- Avatar