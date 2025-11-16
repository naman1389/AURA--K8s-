"""
Kubernetes Tools for MCP Server
Uses Python Kubernetes client instead of raw HTTP requests
"""

import logging
from typing import Dict, List, Optional
from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException

logger = logging.getLogger(__name__)


class KubernetesTools:
    """Tools for interacting with Kubernetes API using official client"""

    def __init__(self):
        """Initialize Kubernetes client (in-cluster or from kubeconfig)"""
        try:
            # Try to load in-cluster config first (running inside pod)
            config.load_incluster_config()
            logger.info("Loaded in-cluster Kubernetes config")
        except config.config_exception.ConfigException:
            # Fall back to kubeconfig from user's machine
            try:
                config.load_kube_config()
                logger.info("Loaded kubeconfig from user machine")
            except config.config_exception.ConfigException as e:
                logger.error(f"Failed to load Kubernetes config: {e}")
                raise

        self.v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.batch_v1 = client.BatchV1Api()

    def get_pod(self, namespace: str, name: str) -> Dict:
        """
        Get pod details
        Returns pod object as dict with key information
        """
        try:
            pod = self.v1.read_namespaced_pod(name, namespace)
            return {
                "name": pod.metadata.name,
                "namespace": pod.metadata.namespace,
                "status": pod.status.phase,
                "node": pod.spec.node_name,
                "containers": [c.name for c in pod.spec.containers],
                "restart_count": sum(
                    cs.restart_count or 0 for cs in pod.status.container_statuses or []
                ),
                "ready": pod.status.conditions[0].status == "True" if pod.status.conditions else False,
                "creation_timestamp": pod.metadata.creation_timestamp.isoformat() if pod.metadata.creation_timestamp else None,
            }
        except ApiException as e:
            logger.error(f"Failed to get pod {namespace}/{name}: {e}")
            return {"error": f"Pod not found: {e.reason}"}

    def get_pod_logs(self, namespace: str, name: str, container: str = None, lines: int = 100) -> str:
        """
        Get pod logs (last N lines)
        Returns log text as string
        """
        try:
            logs = self.v1.read_namespaced_pod_log(
                name,
                namespace,
                container=container,
                tail_lines=lines,
                timestamps=True,
            )
            return logs
        except ApiException as e:
            logger.error(f"Failed to get logs for {namespace}/{name}: {e}")
            return f"Logs not available: {e.reason}"

    def get_events(self, namespace: str, pod_name: str, limit: int = 10) -> List[Dict]:
        """
        Get recent events for a pod
        Returns list of event details
        """
        try:
            events = self.v1.list_namespaced_event(
                namespace,
                field_selector=f"involvedObject.name={pod_name}",
                limit=limit,
            )

            event_list = []
            for event in events.items:
                event_list.append({
                    "message": event.message,
                    "reason": event.reason,
                    "type": event.type,
                    "timestamp": event.last_timestamp.isoformat() if event.last_timestamp else None,
                    "count": event.count,
                })

            return event_list
        except ApiException as e:
            logger.error(f"Failed to get events for {namespace}/{pod_name}: {e}")
            return [{"error": f"Events not available: {e.reason}"}]

    def get_deployment_for_pod(self, namespace: str, pod_name: str) -> Optional[Dict]:
        """
        Get deployment that owns the pod
        Traces through OwnerReferences
        """
        try:
            # Get the pod
            pod = self.v1.read_namespaced_pod(pod_name, namespace)

            # Find owner
            for owner_ref in pod.metadata.owner_references or []:
                if owner_ref.kind == "ReplicaSet":
                    # Get ReplicaSet
                    rs = self.apps_v1.read_namespaced_replica_set(
                        owner_ref.name, namespace
                    )

                    # Find Deployment owner of ReplicaSet
                    for rs_owner in rs.metadata.owner_references or []:
                        if rs_owner.kind == "Deployment":
                            deployment = self.apps_v1.read_namespaced_deployment(
                                rs_owner.name, namespace
                            )
                            return {
                                "name": deployment.metadata.name,
                                "namespace": deployment.metadata.namespace,
                                "replicas": deployment.spec.replicas,
                                "ready_replicas": deployment.status.ready_replicas or 0,
                                "image": deployment.spec.template.spec.containers[0].image
                                if deployment.spec.template.spec.containers
                                else None,
                            }

                elif owner_ref.kind == "Deployment":
                    # Direct deployment owner
                    deployment = self.apps_v1.read_namespaced_deployment(
                        owner_ref.name, namespace
                    )
                    return {
                        "name": deployment.metadata.name,
                        "namespace": deployment.metadata.namespace,
                        "replicas": deployment.spec.replicas,
                        "ready_replicas": deployment.status.ready_replicas or 0,
                        "image": deployment.spec.template.spec.containers[0].image
                        if deployment.spec.template.spec.containers
                        else None,
                    }

            return None

        except ApiException as e:
            logger.error(f"Failed to get deployment for {namespace}/{pod_name}: {e}")
            return None

    def get_node_info(self, node_name: str) -> Optional[Dict]:
        """
        Get information about a node
        """
        try:
            node = self.v1.read_node(node_name)
            return {
                "name": node.metadata.name,
                "status": node.status.conditions[-1].type if node.status.conditions else "Unknown",
                "cpu": node.status.allocatable.get("cpu", "Unknown"),
                "memory": node.status.allocatable.get("memory", "Unknown"),
                "pod_capacity": node.status.allocatable.get("pods", "Unknown"),
            }
        except ApiException as e:
            logger.error(f"Failed to get node info for {node_name}: {e}")
            return None

    def get_pod_resource_usage(self, namespace: str, pod_name: str) -> Optional[Dict]:
        """
        Get current resource usage for a pod
        Requires metrics-server to be installed
        """
        try:
            # Try to use metrics API
            custom_api = client.CustomObjectsApi()
            pod_metrics = custom_api.get_namespaced_custom_object(
                group="metrics.k8s.io",
                version="v1beta1",
                namespace=namespace,
                plural="pods",
                name=pod_name,
            )

            containers = pod_metrics.get("containers", [])
            total_cpu = 0
            total_memory = 0

            for container in containers:
                cpu_str = container.get("usage", {}).get("cpu", "0")
                mem_str = container.get("usage", {}).get("memory", "0")
                
                # Parse CPU (e.g., "100m" -> 100)
                if cpu_str.endswith("m"):
                    total_cpu += int(cpu_str[:-1])
                else:
                    total_cpu += int(float(cpu_str) * 1000)

                # Parse Memory (e.g., "128Mi" -> bytes)
                if mem_str.endswith("Ki"):
                    total_memory += int(mem_str[:-2]) * 1024
                elif mem_str.endswith("Mi"):
                    total_memory += int(mem_str[:-2]) * 1024 * 1024
                elif mem_str.endswith("Gi"):
                    total_memory += int(mem_str[:-2]) * 1024 * 1024 * 1024

            return {
                "cpu_millicores": total_cpu,
                "memory_bytes": total_memory,
                "containers": [c.get("name") for c in containers],
            }

        except ApiException as e:
            logger.warning(f"Failed to get metrics for {namespace}/{pod_name} (metrics-server may not be installed): {e}")
            return None

    def describe_pod(self, namespace: str, pod_name: str) -> str:
        """
        Generate a detailed description of pod and its issues
        Combines pod info, events, and logs
        """
        try:
            pod_info = self.get_pod(namespace, pod_name)
            events = self.get_events(namespace, pod_name, limit=5)
            logs = self.get_pod_logs(namespace, pod_name, lines=20)

            description = f"""
POD DETAILS:
  Name: {pod_info.get('name')}
  Namespace: {pod_info.get('namespace')}
  Status: {pod_info.get('status')}
  Node: {pod_info.get('node')}
  Restart Count: {pod_info.get('restart_count')}
  Ready: {pod_info.get('ready')}
  Created: {pod_info.get('creation_timestamp')}

RECENT EVENTS:
"""
            for event in events[:3]:
                description += f"  [{event.get('type')}] {event.get('reason')}: {event.get('message')}\n"

            description += f"\nRECENT LOGS (last 20 lines):\n{logs}\n"

            return description

        except Exception as e:
            logger.error(f"Failed to describe pod: {e}")
            return f"Failed to describe pod: {e}"

    def get_namespace_resources(self, namespace: str) -> Dict:
        """
        Get overview of resources in a namespace
        """
        try:
            pods = self.v1.list_namespaced_pod(namespace)
            deployments = self.apps_v1.list_namespaced_deployment(namespace)
            statefulsets = self.apps_v1.list_namespaced_stateful_set(namespace)
            services = self.v1.list_namespaced_service(namespace)

            return {
                "pods": len(pods.items),
                "deployments": len(deployments.items),
                "statefulsets": len(statefulsets.items),
                "services": len(services.items),
            }
        except ApiException as e:
            logger.error(f"Failed to get namespace resources for {namespace}: {e}")
            return {"error": f"Failed to get resources: {e.reason}"}
