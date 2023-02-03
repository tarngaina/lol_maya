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
            - MAPGEO: For fixing map crashed while casting a skill has ground indicator.
        - Example: If you want to export modified `yone_base.skl` base on original file, you must have either `riot_yone_base.skl` or `riot.skl` in export location; if you have both of them, `riot_yone_base.skl` will take priority.
2. SKN: 
    - SKN data in Maya scene: 
        - Combined mesh : A single mesh that:
            - Materials and UVs assigned on all faces.
            - All face normals point inward. 
            - Bound with joints as a skin cluster and have weight painted on all vertices.

            ![](https://i.imgur.com/P6S36i0.png)
        - Group meshes: A group of combined mesh.

            ![](https://i.imgur.com/m1dfCgR.png)
    - Read: 
        - `33 22 11 00`: V0, V1, V2, V4
        - SKN import options:
            - Import skeleton: load with SKL as skin cluster.
                - Material that has duplicated name with another joint will be renamed to full lowercase letters. 
                - Example: If there is a joint `Fish`, material `Fish` will be renamed to `fish` in scene.
            - Import mesh separated by material: load SKN as group meshes separated by materials.
        
            ![](https://i.imgur.com/HhoFMLu.png)

    - Write: 
        - To export: 
            - Combined mesh: select the bound mesh -> use export selection.
            - Group meshes: select the group of bound meshes -> use export selection.
        - `33 22 11 00`: V1
        - Limit vertices: 65535.
        - Show/select component on error: 
            - Vertex: 4+ influences vertex, material shared vertex, non UVs assigned vertex.
            - Face: invalid triangulation face, non material assigned face, non UVs assigned face.
3. SKL:
    - SKL data in Maya scene: all joints in scene, it does not matter if ther are bound or not.
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
        - To delete Channels/previous animations before load ANM, change ANM import options to:
        
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
        - A single mesh that: 
            - UVs assigned on all faces.
            - All face normals point inward.
        - Central point: is translation of mesh's transform.
        - Pivot point (SCO only, optional): is translation of an additional joint that bound with mesh.
            
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
![](https://i.imgur.com/w0bxDow.png)
1. Hover mouse on shelf buttons to read tooltip.
2. Explain some buttons:
    - Namespace buttons: Quickly add/remove a temporary namespace on selected objects.
    - Separated mesh button: Separate selected mesh by materials.
    - Fix shared vertices button.
    - Martin UV helper: move selected UVs to specific corner.
    - Update bind pose button: set current pose as bind pose for skin cluster, require: select single joint of the skin cluster.
    - Freeze joints buttons: Freeze/bake selected/all joints rotation.
    - Mirror joint buttons: work great if joints rotations have been frozen.
        - L<->R: mirror rotation of a selected joint startswith `L_` or `R_` to the opposite joint.
        - A<->B: mirror rotation of first selected joint to second selected joint.
    - 4 influences fix button: prune and force max 4 influences on selected skin cluster.
    - Rebind button: Quickly unbind, delete history, rebind selected skin cluster.

### MAPGEO
1. MAPGEO data in Maya scene:
    - A group of meshes that:
        - Materials and UVs assigned on all faces.
        - All face normals point inward.

        ![](https://i.imgur.com/AlmIqQV.png)
    
    - Material:
        - Material names in MAPGEO files or in BIN files are used with `/`, this character can't be used in Maya, so all `/` will be converted to `__` when import, and will be converted back to `/` when export.
        - Example: 
            - In mapgeo or bin: `Maps/KitPieces/Howling_Abyss/Materials/Keep_inst`
            - In Maya: `Maps__KitPieces__Howling_Abyss__Materials__Keep_inst`
        - Material type that used in Maya for MAPGEO by default is Lambert, however, Standard Surface / Arnold's Standard Surface is more advanced to use when export/import materials data through shelf buttons.
        - To import MAPGEO using Standard Surface materials, change MAPGEO import options to:
            
            ![](https://i.imgur.com/QPUqmYZ.png)
    - Layer: 
        - A map must have 8 layers, equal to 8 `set` in Maya.

            ![](https://i.imgur.com/uxlcYsw.png)
        - If an object is assigned to a layer, it will appear on that layer in game.
        - An object can be assigned to multiple layers, if it is assigned on all 8 layers, it will appear on all 8 layers.
        - Layer in Summoner Rift (SR):
            - Layer 1: Base 
            - Layer 2: Inferno
            - Layer 3: Mountain
            - Layer 4: Ocean
            - Layer 5: Cloud
            - Layer 6: Hextech
            - Layer 7: Chemtech
            - Layer 8: Unknown
            - Example in SR: if mesh assigned to `set2` -> that object will appear in layer 2 - Inferno map.
        - Layer in Aram / other maps: objects are always assigned to all 8 layers.

    - Lightmap (Optional):
        - 2nd texture & UVs to store light data in, will blend with main texture (diffuse texture) in game.
        - Lightmap contains two things: Name and UVs data

            ![](https://i.imgur.com/JLKRvQz.png)
        - Name: 
            - If there is no lightmap: The group name can be whatever it is.
            - If there is lightmap: The group name must be lightmap folder path and have `riot_` before it. Example: In Aram, keep this name if you want original lightmap work.
            
                ![](https://i.imgur.com/kRBTsdf.png)

            - Example Lightmap name in Aram: `ASSETS/Maps/Lightmaps/Maps/MapGeometry/Map12/Base/2.dds`
        - UVs data: 2nd UV set of mesh, can be generated by a button on shelf.
        - You don't need to have Lightmap if you can bake light straight into main texture, like Riot did with SR.
    
    - Bucket Grid (crucial):
        - Bucket Grid can only be achieved by exporting base on original MAPGEO file. (aka `riot.mapgeo` method)
        - If you export without Bucket Grid, the modified map will crash League when a skill that has ground indicator casted, example: Miss Fortune E, Gankplank R,...
2. Read:
    - `OEGM`: V5, V6, V7, V9, V11, V12, V13
3. Write:
    - To export: select the group of meshes -> use export selection.
    - `OEGM`: V13
    - Limit vertices for each mesh: 65535.
    - Show/select component on error: 
        - Vertex: material shared vertex, non UVs assigned vertex.
        - Face: invalid triangulation face, non material assigned face, non UVs assigned face.
    - Diffuse UVs must be in 1st UV set; if model uses Lightmap, Lightmap UVs must be in 2nd UV set.
    - Freeze all meshes's transform before you export, because mesh's transform values are not supported in MAPGEO.

        ![](https://i.imgur.com/8eZIWQU.png)
4. Shelf buttons:

    ![](https://i.imgur.com/dRiK2Mt.png)

    Explain buttons from left to right:
    - Rename path of selected objects with input prefix.
    - Rename selected objects with input prefix. 
    - Toggle on/off all layers on selected mesh.
    - Toggle on/off layer 1 - 8 on selected mesh.
    - Select all faces of all meshes that have same assigned material as selected face.
    - Set all black emissions weight to 0.
    - Fix shared vertices on all meshes in scene.
    - Import `materials.py`: 
        - Read `materials.py` file to import textures.
        - `assets` folder must be in same location as `materials.py`.
    - Export `materials.json`: export all materials of selected meshes to a json file; all textures will be copied to same export location; if nothing get selected, this button export all materials in scene instead; `materials.json` can be read by `Avatar (made by Killery)` to convert back to `materials.py`.
    - Export `materials.py`: export all materials of selected meshes to a py file; all textures will be copied to same export location; if nothing get selected, this button export all materials in scene instead; only diffuse textures will be exported and only export into `default_env` shader.
    - Extra: League shader <-> Maya shader:
        - Lambert / Other materials:
            - Diffuse Texture -> Color / Transparency
        - Standard Surface / Arnold's Standard Surface:
            - Diffuse Texture <-> Base
            - Glow Texture <-> Base, Glow Color <-> Glow
            - Mask Color <-> Coat, Mask Texture <-> Coat Normal
            - Emissive Texture <-> Emissive Color <-> Emission
    - Generate Lightmap UVs on 2nd UV set of selected objects.
    - Delete Lightmap UVs and 2nd UV set of selected objects.
    - Bake texture: bake textures with [Arnold](https://arnoldrenderer.com/download/) on selected objects.
        
        ![](https://i.imgur.com/4bEXne8.png)
        
        - Output: Location of output baked textures.
        - No diffuse:
            - On: Bake only light - use 2nd UV set of selected objects and default material `standardSurface1` to bake.
            - Off: Bake with diffuse - use 1st UV set of selected objects and their own materials to bake.
        - Quality: You will want High quality bake for diffuse and Low quality bake for lightmap.
        - Resolution: Resolution of baked textures; integer input, should be 256, 512, 1024,...

### External Links:

- [LeagueFileTranlastor](https://github.com/LoL-Fantome/LeagueFileTranslator)

- [LeagueToolKit](https://github.com/LoL-Fantome/LeagueToolkit)

- [LoL2Blender](https://github.com/WorldSEnder/LoL2Blender)

- [Maya SDK Reference](https://help.autodesk.com/cloudhelp/2023/ENU/Maya-SDK/cpp_ref/modules.html)

- [Arnold](https://docs.arnoldrenderer.com/display/A5AFMUG/Arnold+for+Maya+User+Guide)

- [Avatar](https://github.com/TheKillerey/Avatar)
