# FLUX.1 Schnell (and others) Manifold Functions for Open WebUI

## Overview

This repository contains the implementation of the FLUX.1 Schnell Manifold Function, a utility designed for Black Forest Lab Image Generation Models. The function interacts with various API providers to generate images based on user inputs directly into the Open WebUI chat window.

## Features

- **API Integration**: Supports multiple API providers including Hugging Face, Replicate, and Together.
- **Environment Configuration**: Utilizes environment variables for API keys and base URLs.

## Requirements

- Python 3.7+
- `pydantic`
- `requests`

## Installation

1. Download the source and import into Open WebUI functions interface.


2. Install via official Open WebUI community functions repository:

[flux_schnell](https://openwebui.com/f/bgeneto/flux_schnell)

## Environment Variables

Set the following environment variables in your environment:

- `FLUX_SCHNELL_API_BASE_URL`: Base URL for the API.
- `FLUX_SCHNELL_API_KEY`: Your API Key for Flux.1 Schnell.

Example:
```sh
export FLUX_SCHNELL_API_BASE_URL="https://api.replicate.com/v1/models/black-forest-labs/flux-schnell/predictions"
export FLUX_SCHNELL_API_KEY="your_replicate_api_key_here"

export FLUX_SCHNELL_API_BASE_URL="https://api.together.xyz/v1/images/generations"
export FLUX_SCHNELL_API_KEY="your_together_api_key_here"
```

Screenshots:
![image](https://github.com/user-attachments/assets/9219f626-e269-4906-a913-133e4113c10e)
