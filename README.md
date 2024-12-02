# Vertical Federated Learning with Client Dropout

This repository is a modified version of the original implementation of Vertical Federated Learning (VFL) by Kang Wei et al. Our changes focus on investigating the behavior of VFL under client dropout scenarios.

## 📄 **Credit**

The original implementation was authored by:

- Kang Wei  
- Jun Li  
- Chuan Ma  
- Ming Ding  
- Sha Wei  
- Fan Wu  
- Guihai Chen  
- Thilina Ranbaduge  

The corresponding paper can be found here: [arXiv:2202.04309](https://arxiv.org/abs/2202.04309)  
The original GitHub repository is available here: [Vertical_FL](https://github.com/AdamWei-boop/Vertical_FL)

## 🛠️ **Changes**

This repository has been modified to suit our course project, which investigates the impact of client dropout in Vertical Federated Learning.  
Key changes include:

1. **Removed Features**:
   - Differential Privacy (DP) implementation.
   - Contribution calculation logic.

2. **Added Features**:
   - Logic for handling **active clients**.
   - A toggleable variable named `active_clients` to specify which clients are currently participating in the training process.

## 🚀 **Getting Started**

### Prerequisites

Ensure you have `pipenv` installed to manage the Python environment.

### Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
