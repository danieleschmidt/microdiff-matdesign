#!/usr/bin/env python3
"""
Software Bill of Materials (SBOM) Generator for MicroDiff-MatDesign

Generates comprehensive SBOM in multiple formats (SPDX, CycloneDX, SWID)
for compliance with supply chain security requirements.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import pkg_resources
except ImportError:
    pkg_resources = None

try:
    import toml
except ImportError:
    toml = None


class SBOMGenerator:
    """Generate Software Bill of Materials in various formats."""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.project_info = self._load_project_info()
        
    def _load_project_info(self) -> Dict[str, Any]:
        """Load project information from pyproject.toml."""
        pyproject_path = self.project_root / "pyproject.toml"
        
        if not toml or not pyproject_path.exists():
            return {
                "name": "microdiff-matdesign",
                "version": "0.1.0",
                "description": "Diffusion model framework for inverse material design"
            }
        
        with open(pyproject_path, "r") as f:
            data = toml.load(f)
            
        project = data.get("project", {})
        return {
            "name": project.get("name", "microdiff-matdesign"),
            "version": self._get_version(),
            "description": project.get("description", ""),
            "authors": project.get("authors", []),
            "license": project.get("license", {}),
            "homepage": project.get("urls", {}).get("Homepage", ""),
            "repository": project.get("urls", {}).get("Repository", "")
        }
    
    def _get_version(self) -> str:
        """Get version from git or fallback."""
        try:
            result = subprocess.run(
                ["git", "describe", "--tags", "--dirty"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "0.1.0-dev"
    
    def _get_python_packages(self) -> List[Dict[str, Any]]:
        """Get installed Python packages and their versions."""
        packages = []
        
        if pkg_resources:
            for dist in pkg_resources.working_set:
                packages.append({
                    "name": dist.project_name,
                    "version": dist.version,
                    "type": "python-package",
                    "supplier": "PyPI",
                    "location": dist.location,
                    "requires": [str(req) for req in dist.requires()]
                })
        else:
            # Fallback: try pip list
            try:
                result = subprocess.run(
                    ["pip", "list", "--format=json"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                pip_packages = json.loads(result.stdout)
                
                for pkg in pip_packages:
                    packages.append({
                        "name": pkg["name"],
                        "version": pkg["version"],
                        "type": "python-package",
                        "supplier": "PyPI"
                    })
            except (subprocess.CalledProcessError, json.JSONDecodeError):
                pass
        
        return packages
    
    def _get_system_packages(self) -> List[Dict[str, Any]]:
        """Get system packages (Ubuntu/Debian)."""
        packages = []
        
        try:
            result = subprocess.run(
                ["dpkg-query", "-W", "-f=${Package}\t${Version}\t${Architecture}\n"],
                capture_output=True,
                text=True,
                check=True
            )
            
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        packages.append({
                            "name": parts[0],
                            "version": parts[1],
                            "architecture": parts[2] if len(parts) > 2 else "unknown",
                            "type": "deb-package",
                            "supplier": "Ubuntu/Debian"
                        })
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        return packages
    
    def _get_docker_info(self) -> Optional[Dict[str, Any]]:
        """Get Docker base image information."""
        dockerfile_path = self.project_root / "Dockerfile"
        
        if not dockerfile_path.exists():
            return None
        
        base_images = []
        
        with open(dockerfile_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("FROM "):
                    image = line.replace("FROM ", "").split(" as ")[0]
                    base_images.append(image)
        
        return {
            "base_images": base_images,
            "dockerfile_path": str(dockerfile_path)
        }
    
    def _get_git_info(self) -> Dict[str, Any]:
        """Get Git repository information."""
        git_info = {}
        
        try:
            # Get commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            git_info["commit_hash"] = result.stdout.strip()
            
            # Get remote URL
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            git_info["remote_url"] = result.stdout.strip()
            
            # Get branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            git_info["branch"] = result.stdout.strip()
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        return git_info
    
    def generate_spdx(self) -> Dict[str, Any]:
        """Generate SBOM in SPDX format."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        sbom = {
            "spdxVersion": "SPDX-2.3",
            "dataLicense": "CC0-1.0",
            "SPDXID": "SPDXRef-DOCUMENT",
            "name": f"{self.project_info['name']}-{self.project_info['version']}",
            "documentNamespace": f"https://github.com/danieleschmidt/microdiff-matdesign/{self.project_info['version']}",
            "creationInfo": {
                "created": timestamp,
                "creators": ["Tool: microdiff-sbom-generator"],
                "licenseListVersion": "3.19"
            },
            "packages": [],
            "relationships": []
        }
        
        # Add main project package
        main_package = {
            "SPDXID": "SPDXRef-Package-microdiff-matdesign",
            "name": self.project_info["name"],
            "versionInfo": self.project_info["version"],
            "downloadLocation": self.project_info.get("repository", "NOASSERTION"),
            "filesAnalyzed": False,
            "copyrightText": "NOASSERTION",
            "supplier": "Organization: Terragon Labs"
        }
        
        if self.project_info.get("license", {}).get("text"):
            main_package["licenseConcluded"] = self.project_info["license"]["text"]
        
        sbom["packages"].append(main_package)
        
        # Add Python packages
        for pkg in self._get_python_packages():
            spdx_id = f"SPDXRef-Package-{pkg['name'].replace('-', '').replace('_', '')}"
            
            package = {
                "SPDXID": spdx_id,
                "name": pkg["name"],
                "versionInfo": pkg["version"],
                "downloadLocation": f"https://pypi.org/project/{pkg['name']}/",
                "filesAnalyzed": False,
                "copyrightText": "NOASSERTION",
                "supplier": "Organization: Python Package Index"
            }
            
            sbom["packages"].append(package)
            
            # Add relationship
            sbom["relationships"].append({
                "spdxElementId": "SPDXRef-Package-microdiff-matdesign",
                "relatedSpdxElement": spdx_id,
                "relationshipType": "DEPENDS_ON"
            })
        
        # Add system packages
        for pkg in self._get_system_packages():
            spdx_id = f"SPDXRef-System-{pkg['name'].replace('-', '').replace('_', '')}"
            
            package = {
                "SPDXID": spdx_id,
                "name": pkg["name"],
                "versionInfo": pkg["version"],
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "copyrightText": "NOASSERTION",
                "supplier": f"Organization: {pkg['supplier']}"
            }
            
            sbom["packages"].append(package)
        
        return sbom
    
    def generate_cyclonedx(self) -> Dict[str, Any]:
        """Generate SBOM in CycloneDX format."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.4",
            "serialNumber": f"urn:uuid:{self.project_info['name']}-{timestamp}",
            "version": 1,
            "metadata": {
                "timestamp": timestamp,
                "tools": [{
                    "vendor": "Terragon Labs",
                    "name": "microdiff-sbom-generator",
                    "version": "1.0.0"
                }],
                "component": {
                    "type": "application",
                    "bom-ref": "microdiff-matdesign",
                    "name": self.project_info["name"],
                    "version": self.project_info["version"],
                    "description": self.project_info["description"]
                }
            },
            "components": [],
            "dependencies": []
        }
        
        # Add repository info if available
        git_info = self._get_git_info()
        if git_info.get("remote_url"):
            sbom["metadata"]["component"]["externalReferences"] = [{
                "type": "vcs",
                "url": git_info["remote_url"]
            }]
        
        # Add Python components
        main_deps = []
        
        for pkg in self._get_python_packages():
            bom_ref = f"pkg:pypi/{pkg['name']}@{pkg['version']}"
            
            component = {
                "type": "library",
                "bom-ref": bom_ref,
                "name": pkg["name"],
                "version": pkg["version"],
                "purl": bom_ref,
                "scope": "required"
            }
            
            sbom["components"].append(component)
            main_deps.append(bom_ref)
        
        # Add main dependency
        sbom["dependencies"].append({
            "ref": "microdiff-matdesign",
            "dependsOn": main_deps
        })
        
        # Add Docker info if available
        docker_info = self._get_docker_info()
        if docker_info:
            for base_image in docker_info["base_images"]:
                if ":" in base_image:
                    name, version = base_image.rsplit(":", 1)
                else:
                    name, version = base_image, "latest"
                
                component = {
                    "type": "container",
                    "bom-ref": f"docker:{base_image}",
                    "name": name,
                    "version": version,
                    "scope": "required"
                }
                
                sbom["components"].append(component)
        
        return sbom
    
    def generate_simple_json(self) -> Dict[str, Any]:
        """Generate simple JSON SBOM for internal use."""
        return {
            "project": self.project_info,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "git": self._get_git_info(),
            "python_packages": self._get_python_packages(),
            "system_packages": self._get_system_packages(),
            "docker": self._get_docker_info()
        }
    
    def save_sbom(self, sbom_data: Dict[str, Any], output_path: Path, format_type: str):
        """Save SBOM to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(sbom_data, f, indent=2, sort_keys=True)
        
        print(f"Generated {format_type} SBOM: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Software Bill of Materials for MicroDiff-MatDesign"
    )
    
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sbom"),
        help="Output directory for SBOM files (default: ./sbom)"
    )
    
    parser.add_argument(
        "--format",
        choices=["spdx", "cyclonedx", "json", "all"],
        default="all",
        help="SBOM format to generate (default: all)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Project root: {args.project_root}")
        print(f"Output directory: {args.output_dir}")
    
    generator = SBOMGenerator(args.project_root)
    
    if args.format in ["spdx", "all"]:
        spdx_data = generator.generate_spdx()
        generator.save_sbom(
            spdx_data,
            args.output_dir / "microdiff-matdesign.spdx.json",
            "SPDX"
        )
    
    if args.format in ["cyclonedx", "all"]:
        cyclonedx_data = generator.generate_cyclonedx()
        generator.save_sbom(
            cyclonedx_data,
            args.output_dir / "microdiff-matdesign.cyclonedx.json",
            "CycloneDX"
        )
    
    if args.format in ["json", "all"]:
        json_data = generator.generate_simple_json()
        generator.save_sbom(
            json_data,
            args.output_dir / "microdiff-matdesign.sbom.json",
            "JSON"
        )
    
    print("SBOM generation completed successfully!")


if __name__ == "__main__":
    main()
