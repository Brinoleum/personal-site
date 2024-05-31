---
title: "Raspberry Pi LLM"
date: 2024-05-30T19:31:17-07:00
---

Recently, Picovoice announced the [release of PicoLLM](https://picovoice.ai/blog/picollm-local-llm-platform/), a lightweight local inferencing platform that relies on compressing the weights of an LLM to optimize for speed and limited computing power. In particular, models can be run on smaller devices without the full power of a GPU. Having recently purchased a Raspberry Pi 5 but having no idea what kinds of projects I wanted to run, this seemed to be the perfect opportunity for me to try it out. Previous attempts at similar LLM projects on limited computers fizzled out, mostly due to constraints on cost (>100$ just for a M.2 AI coprocessor??? Not to mention that Google Coral, while ostensibly cheaper, is basically abandoned from what I can tell) and hardware (read: I got a very nice case with integrated fan from Canakit and trying to fit the M.2 HAT in would have made active cooling a challenge, and, more importantly, ruined the cool aesthetic).

Following the instructions on getting started, I set up my Python environment in Anaconda and installed the PicoLLM package on the RasPi. I also downloaded the model weights provided by Picovoice for the Phi2 model and uploaded to the Pi, which took a hot minute even for the comparatively small 1.7GB weights. Once all that was done, I quickly wrote up a Python script to prompt the model and off to the races we went!

To be entirely honest, it was quite slow to provide a response, which wasn't unexpected given the processing power of the Pi's internals. What was more surprising was that when I killed the process in the terminal, the printed response was flushed to stdout, which showed that the inference was complete - only the cleanup code from the Pico API was lagging behind. Clearly, I'm not going to see results streaming in quite as quickly as I would with the ChatGPT interface, but for what it is, the overall capability is more than okay.

So far, I haven't really dug into more interesting things to do with this setup quite yet, but I still think it's looking quite promising so far. I've got some other project ideas cooking up, so stay tuned!