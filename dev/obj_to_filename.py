import simulacra as si

if __name__ == '__main__':
    obj = si.Specification('John', foo = 'bar', baz = 'bo')

    print(obj.info())

    print(si.utils.obj_to_filename(obj, [
        # ('__class__', '__name__'),
        'name',
        'foo',
        ('baz', lambda x: x.upper())
    ]))
