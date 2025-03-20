"""Instrumentation common."""
import os
import socket

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.resources import DEPLOYMENT_ENVIRONMENT, HOST_NAME, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as OTLPSpanGrpcExporter

from aliyun.instrumentation.context import context

from pai_rag.trace.trace_config import TraceConfig
set_custom_attributes = context.set_custom_attributes

"""Llama-index instrumentor."""
from aliyun.instrumentation.llama_index import AliyunLlamaIndexInstrumentor

from loguru import logger


def init_opentelemetry(config: TraceConfig,
                       service_app_name='',
                       service_version='',
                       service_id='',
                       deployment_environment='',
                       service_owner_id='',
                       service_owner_sub_id=''):
    """Init opentelemetry."""
    # set up instrumentation
    grpc_endpoint = config.endpoint
    token = config.token
    service_name = config.service_name
    
    attributes = {SERVICE_NAME: service_name, HOST_NAME: socket.gethostname()}

    if not service_app_name:
        logger.error("serice_app_name not provided in trace config.")
        raise ValueError('service_app_name must be provided!')
    
    if not token:
        logger.error("token not provided in trace config.")
        raise ValueError('token must be provided!')
    
    attributes['service.app.name'] = service_app_name

    if service_version:
        attributes[SERVICE_VERSION] = service_version

    if service_id:
        attributes['service.id'] = service_id

    if deployment_environment:
        attributes[DEPLOYMENT_ENVIRONMENT] = deployment_environment

    if service_owner_id:
        attributes['service.owner.id'] = service_owner_id

    if service_owner_sub_id:
        attributes['service.owner.sub_id'] = service_owner_sub_id

    resource = Resource(attributes=attributes)
    exporter = OTLPSpanGrpcExporter(endpoint=grpc_endpoint, headers=(f"Authentication={token}"))
    if os.getenv('DEBUG_INSTRUMENTATION'):
        exporter = ConsoleSpanExporter()
        logger.info(f'in DEBUG_INSTRUMENTATION mode, exporter: {exporter}')
    span_processor = BatchSpanProcessor(exporter)
    trace_provider = TracerProvider(resource=resource, active_span_processor=span_processor)
    trace.set_tracer_provider(trace_provider)

    AliyunLlamaIndexInstrumentor().instrument()


def get_tracer():
    """Get tracer."""
    #return trace.get_tracer(__name__, tracer_provider=trace.get_tracer_provider())
    return trace.get_tracer(__name__)


