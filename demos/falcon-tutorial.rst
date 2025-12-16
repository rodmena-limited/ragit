Tutorial (WSGI)
===============

In this tutorial we'll walk through building an API for a simple image sharing
service. Along the way, we'll discuss Falcon's major features and introduce
the terminology used by the framework.

First Steps
-----------

Install Falcon inside a virtualenv:

.. code:: bash

    $ pip install falcon

Create your app entry point ``app.py``:

.. code:: python

    import falcon

    app = application = falcon.App()

This code creates your WSGI application. A WSGI application is just a callable
with a well-defined signature so you can host it with any WSGI server.

Hosting Your App
----------------

Use Gunicorn to run your app:

.. code:: bash

    $ pip install gunicorn
    $ gunicorn --reload look.app

For Windows, use Waitress:

.. code:: bash

    $ pip install waitress
    $ waitress-serve --port=8000 look.app:app

Creating Resources
------------------

Central to Falcon is the concept of a "resource". Resources are simply all
the things in your API that can be accessed by a URL.

Falcon uses Python classes to represent resources. Each class acts as a
controller that converts incoming requests into actions and composes responses.

Example resource class:

.. code:: python

    import json
    import falcon


    class Resource:

        def on_get(self, req, resp):
            doc = {
                'images': [
                    {
                        'href': '/images/1eaf6ef1-7f2d-4ecc-a8d5-6e8adba7cc0e.png'
                    }
                ]
            }
            resp.text = json.dumps(doc, ensure_ascii=False)
            resp.status = falcon.HTTP_200

For any HTTP method you want to support, add an ``on_*()`` method to the class,
where ``*`` is the HTTP method lowercased (e.g., ``on_get()``, ``on_post()``,
``on_put()``, ``on_delete()``).

Wire up the resource:

.. code:: python

    import falcon
    from .images import Resource

    app = application = falcon.App()

    images = Resource()
    app.add_route('/images', images)

Request and Response Objects
----------------------------

Each responder receives a ``Request`` object for reading headers, query params,
and body. Also receives a ``Response`` object for setting status, headers, and body.

POST Example with file upload:

.. code:: python

    import io
    import os
    import uuid
    import mimetypes
    import falcon


    class Resource:

        _CHUNK_SIZE_BYTES = 4096

        def __init__(self, storage_path):
            self._storage_path = storage_path

        def on_post(self, req, resp):
            ext = mimetypes.guess_extension(req.content_type)
            name = '{uuid}{ext}'.format(uuid=uuid.uuid4(), ext=ext)
            image_path = os.path.join(self._storage_path, name)

            with io.open(image_path, 'wb') as image_file:
                while True:
                    chunk = req.stream.read(self._CHUNK_SIZE_BYTES)
                    if not chunk:
                        break
                    image_file.write(chunk)

            resp.status = falcon.HTTP_201
            resp.location = '/images/' + name

Status codes: Use ``falcon.HTTP_200``, ``falcon.HTTP_201``, ``falcon.HTTP_404``, etc.
Or aliases like ``falcon.HTTP_OK``, ``falcon.HTTP_CREATED``, ``falcon.HTTP_NOT_FOUND``.

Serving Files
-------------

Return files by setting ``resp.stream`` and ``resp.content_length``:

.. code:: python

    class Item:

        def __init__(self, image_store):
            self._image_store = image_store

        def on_get(self, req, resp, name):
            resp.content_type = mimetypes.guess_type(name)[0]
            resp.stream, resp.content_length = self._image_store.open(name)

Route with URI parameters:

.. code:: python

    app.add_route('/images/{name}', Item(image_store))

The ``{name}`` parameter is passed to ``on_get(self, req, resp, name)``.

Query Strings
-------------

Use ``req.get_param()`` or ``req.get_param_as_int()`` to read query parameters:

.. code:: python

    def on_get(self, req, resp):
        max_size = req.get_param_as_int("maxsize", min_value=1, default=-1)
        limit = req.get_param_as_int("limit", default=10)

Hooks
-----

Use ``@falcon.before`` decorator to run code before a responder:

.. code:: python

    ALLOWED_IMAGE_TYPES = (
        'image/gif',
        'image/jpeg',
        'image/png',
    )

    def validate_image_type(req, resp, resource, params):
        if req.content_type not in ALLOWED_IMAGE_TYPES:
            msg = 'Image type not allowed. Must be PNG, JPEG, or GIF'
            raise falcon.HTTPBadRequest(title='Bad request', description=msg)

    class Collection:
        @falcon.before(validate_image_type)
        def on_post(self, req, resp):
            # ... handle POST

Apply hooks to entire resource class:

.. code:: python

    @falcon.before(extract_project_id)
    class Message:
        pass

Error Handling
--------------

Raise Falcon's built-in HTTP errors:

.. code:: python

    class Item:

        def on_get(self, req, resp, name):
            try:
                resp.stream, resp.content_length = self._image_store.open(name)
            except OSError:
                raise falcon.HTTPNotFound()

Common errors:
- ``falcon.HTTPBadRequest(title='...', description='...')``
- ``falcon.HTTPNotFound()``
- ``falcon.HTTPUnauthorized()``
- ``falcon.HTTPForbidden()``
- ``falcon.HTTPInternalServerError()``

Complete Example App
--------------------

.. code:: python

    import json
    import falcon


    class HelloResource:
        def on_get(self, req, resp):
            resp.text = json.dumps({'message': 'Hello, World!'})
            resp.status = falcon.HTTP_200


    class ItemResource:
        def __init__(self):
            self.items = {}

        def on_get(self, req, resp, item_id):
            if item_id not in self.items:
                raise falcon.HTTPNotFound()
            resp.text = json.dumps(self.items[item_id])

        def on_put(self, req, resp, item_id):
            data = req.media
            self.items[item_id] = data
            resp.status = falcon.HTTP_201

        def on_delete(self, req, resp, item_id):
            if item_id not in self.items:
                raise falcon.HTTPNotFound()
            del self.items[item_id]
            resp.status = falcon.HTTP_204


    app = falcon.App()
    app.add_route('/hello', HelloResource())
    app.add_route('/items/{item_id}', ItemResource())

Run with: ``gunicorn app:app``

Testing
-------

Use ``falcon.testing.TestClient`` for testing:

.. code:: python

    import pytest
    from falcon import testing
    from app import app

    @pytest.fixture
    def client():
        return testing.TestClient(app)

    def test_hello(client):
        response = client.simulate_get('/hello')
        assert response.status == falcon.HTTP_OK
        assert response.json == {'message': 'Hello, World!'}
