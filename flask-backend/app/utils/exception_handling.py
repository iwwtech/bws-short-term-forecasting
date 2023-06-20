from flask import abort, make_response
from marshmallow.exceptions import ValidationError
import sys
import traceback


class BadRequestException(Exception):
    pass


class DataProcessingException(Exception):
    pass


class ResourceNotFoundException(Exception):
    pass


class SaveResourceException(Exception):
    pass


class CredentialsMissingException(Exception):
    pass


class MethodNotAllowedException(Exception):
    pass


class NotImplementedException(Exception):
    pass


class DeleteResourceException(Exception):
    pass

class OrionCommunicationException(Exception):
    pass


def is_known_exception(exception):
    return isinstance(
        exception,
        (
            BadRequestException,
            CredentialsMissingException,
            DataProcessingException,
            DeleteResourceException,
            MethodNotAllowedException,
            NotImplementedException,
            OrionCommunicationException,
            ResourceNotFoundException,
            SaveResourceException,
            ValidationError,
        ),
    )


def make_error_response(e):
    """
    Creates a non-2xx response from a given exception.
    :param e: The exception that was catched. Its message is returned as "msg"
    """
    response_data = {"msg": str(e)}

    if isinstance(e, BadRequestException) or isinstance(e, ValidationError):
        return make_response(response_data, 400)

    if isinstance(e, CredentialsMissingException):
        return make_response(response_data, 401)

    if isinstance(e, ResourceNotFoundException):
        return make_response(response_data, 404)

    if isinstance(e, MethodNotAllowedException):
        return make_response(response_data, 405)

    if (
        isinstance(e, SaveResourceException)
        or isinstance(e, DataProcessingException)
        or isinstance(e, DeleteResourceException)
        or isinstance(e, OrionCommunicationException)
    ):
        return make_response(response_data, 500)

    if isinstance(e, NotImplementedException):
        return make_response(response_data, 405)

    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    abort(400)
