# Claude.md

# Rules to follow:
This file provides guidance to Claude Code when working with code in this repository.

## Rules to Follow:
1. ALWAYS write secure best practice Python code.
2. Always try to write as lean as possible code. Don't blow up the repo. 
4 Iterate function based on test results
5. MOVE Test scripts to the tests folder if they are not already there and ensure that they could be reused for later Tests for code coverage or reruns.
6. ALWAYS commit after each new function is added to our codebase
7. Ensure that you are using uv for isolating environments and packagemanagement
8. Use tree command for project structure. If tree comand not exist install it with command: brew install tree
9. For new and open git issues which should be implemented create first a new branch and work in this branch
10. Ensure that always if a issue is completed pull requests are created.
11. Create a tmp folder for development. And create a scratchpad.md file in this folder to chronologically document the development process.
12. Give the user after each finished step a short advise how to test your implementation. 
13. Always update or create the docs/usage.md file with the newly changed functionality to know how to use the actual implementation.
14. Absolut important keep the repo lean and clean, don't add unnecessary files, don't overengineer.
15. If the tasks has to do to interact with models, make yourself always comfortable with the documentation of the nnsights library, before you start to implement.
16. All development should be done first on small toy models openai-community/gpt2. 
17. Don't use the remote ndif server at the moment. Use the local model by utilising nnsights library.
18. USe Playwright for testing the frontend application.

# Design Decisions:
We haveily use nnsigts library for all interpretabilit and model insight tasks:
- Documentaiton could be found here: https://nnsight.net/documentation/
- Feature overview: https://nnsight.net/features/
- Tutorials: https://nnsight.net/tutorials/
- Github: https://github.com/ndif-team/nnsight
