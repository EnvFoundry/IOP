# IOP 0.17.1

## Research environment contents

Research OS connectors include publisher-authored English and Chinese descriptions and package-relative locations for Skills, instrument guides and configuration. The same declaration ships in the Python distribution and standalone Connect archive. Hosts resolve resource locations after installation; existing laboratory configuration and accepted experiments retain their owners.

Research OS 连接包提供中英文说明，以及 Skills、仪器指南和配置的包内位置；Python 发行包与独立 Connect 包包含同一声明。宿主按实际安装位置解析资源，已有实验室配置和已接收实验保持原有归属。

## Concise English guidance

- One working flow: read the current laboratory, run the task, read the result and continue. The default Skill is about two thirds shorter; apparatus knowledge and methods are read only when needed.
- A single English task entry covers deployment, connection, platform setup, measurement, research and continuation. The former translated entry links to it.
- Setup, apparatus knowledge and research guides remove repeated stages and checklists. They retain model-specific position referencing, measured response, uncertainty and experimental design where the task needs them.
- Shorter MCP instructions and prompts reuse the same guides. Tool names, parameters and experiment formats are unchanged.
- Upkeep tests use explicit completion and a controlled retry clock instead of assuming HTTP calls finish within a short interval.

Includes IOP Connect 1.11.1.
