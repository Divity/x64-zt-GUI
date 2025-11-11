# x64 ZoneTool
A fastfile unlinker and linker for various x64 Call of Duty titles. 

- If you are interested in porting maps or assets from IW3/4/5, check <b>[Aurora's Map Porting IW3/4/5 -> H1](https://docs.auroramod.dev/map-porting-iw5)</b>
- If you are interested in porting maps or assets from newer Call of Duty games like H1, S1, H1, H2, or IW7, between each other, check <b>[Aurora's Map Porting (S1 <-> H1 <-> H2)](https://docs.auroramod.dev/map-porting-s1)</b>


## Gui 
Imgui Based Gui utilizing MicaBlur for the background 

- Dump Tab
<img width="1188" height="789" alt="image" src="https://github.com/user-attachments/assets/c4b9d374-f560-466e-b422-e59a5b4094bf" />

- Build Tab
<img width="1185" height="793" alt="image" src="https://github.com/user-attachments/assets/31432116-a505-4397-ba99-51988394098f" />


- Condole Tab
<img width="1177" height="794" alt="image" src="https://github.com/user-attachments/assets/1abfc4fa-1298-4be4-ae32-5fa22877f863" />


(errors only)<img width="1183" height="796" alt="image" src="https://github.com/user-attachments/assets/0a4ceea8-c41d-4376-bbaf-23282adb3917" />

- CSV Editor

<img width="1188" height="779" alt="image" src="https://github.com/user-attachments/assets/b91f893c-853e-4149-8a54-956cae68afe3" />



## Supported Games
* **IW6** (*Call of Duty: Ghosts*)
* **S1** (*Call of Duty: Advanced Warfare*)
* **T7** (*Call of Duty: Black Ops 3*) ***[dumping only]***
* **H1** (*Call of Duty: Modern Warfare Remastered*)
* **H2** (*Call of Duty: Modern Warfare 2 Campaign Remastered*)
* **IW7** (*Call of Duty: Infinite Warfare*) ***[no custom maps]***

## How to use
Check out the [Aurora Zonetool Basics](https://docs.auroramod.dev/zonetool-basics) for useful guides & information on how to port maps and use zonetool.

## Commands
* `loadzone <zone>`: Loads a zone
* `unloadzones`: Unloads zones
* `verifyzone <zone>`: Lists assets in a zone
* `dumpzone <zone>`: Dumps a zone
* `dumpzone <target game> <zone> <asset filter>`: Dumps a zone converting assets for a specific game
* `dumpasset <type> <name>`: Dumps a single assset
* `dumpmap <map>`: Dumps all required assets for a map
* `dumpmap <target game> <map> <asset filter> <skip common>`: Dumps and converts all required assets for a map

  ### Definitions
  * `asset filter`: A filter specifying all the asset types that should be dumped, if not specified or empty it will dump all asset types.
  Asset types are separated by **commas**, **`_`** indicates and empty filter.   
    * Example: `dumpzone h1 mp_clowntown3 sound,material,techset,rawfile`
    * Example: `dumpmap h1 mp_clowntown3 _ true`
  * `skip common`: Skips common zones when dumping a map, can be `true` or `false`.
  * `target game`: The game to convert the assets to.

## Conversion support
The conversions for how assets can translate is showed on a table below:

- ✔️ = Fully supported
- ⚠️ = Partial (experimental)
- ❌ = Not supported

|            | **IW6** | **S1** | **H1** | **H2** | **T7** | **IW7** |
| ---------- | ------ | ------ | ------ | ------ | ------ | ------ |
| **IW6**    | ✔️     | ❌     | ✔️     | ✔️     | ❌     | ✔️     |
| **S1**     | ❌     | ✔️     | ✔️     | ✔️     | ❌     | ❌     |
| **H1**     | ❌     | ⚠️     | ✔️     | ✔️     | ❌     | ⚠️     |
| **H2**     | ❌     | ❌     | ✔️     | ✔️     | ❌     | ❌     |
| **T7**     | ❌     | ❌     | ⚠️     | ❌     | ❌ | ⚠️     |
| **IW7**    | ❌     | ❌     | ✔️     | ❌     | ❌     | ✔️ |  
