/*!
 * github-buttons v2.14.0
 * (c) 2020 なつき
 * @license BSD-2-Clause
 */
(function () {
    'use strict';

    var document = window.document;

    var location = document.location;

    var Math = window.Math;

    var HTMLElement = window.HTMLElement;

    var XMLHttpRequest = window.XMLHttpRequest;

    var buttonClass = 'github-button';

    var iframeURL = 'https://' + (/* istanbul ignore next */  'buttons.github.io') + '/buttons.html';

    var domain = 'github.com';

    var apiBaseURL = 'https://api.' + domain;

    var useXHR = XMLHttpRequest && 'prototype' in XMLHttpRequest && 'withCredentials' in XMLHttpRequest.prototype;

    var useShadowDOM = useXHR && HTMLElement && 'attachShadow' in HTMLElement.prototype && !('prototype' in HTMLElement.prototype.attachShadow);

    var stringify = function (obj, sep, eq, encodeURIComponent) {
        if (sep == null) {
            sep = '&';
        }
        if (eq == null) {
            eq = '=';
        }
        if (encodeURIComponent == null) {
            encodeURIComponent = window.encodeURIComponent;
        }
        var params = [];
        for (var name in obj) {
            var value = obj[name];
            if (value != null) {
                params.push(encodeURIComponent(name) + eq + encodeURIComponent(value));
            }
        }
        return params.join(sep)
    };

    var parse = function (str, sep, eq, decodeURIComponent) {
        if (sep == null) {
            sep = '&';
        }
        if (eq == null) {
            eq = '=';
        }
        if (decodeURIComponent == null) {
            decodeURIComponent = window.decodeURIComponent;
        }
        var obj = {};
        var params = str.split(sep);
        for (var i = 0, len = params.length; i < len; i++) {
            var entry = params[i];
            if (entry !== '') {
                var ref = entry.split(eq);
                obj[decodeURIComponent(ref[0])] = (ref[1] != null ? decodeURIComponent(ref.slice(1).join(eq)) : undefined);
            }
        }
        return obj
    };

    var onEvent = function (target, eventName, func) {
        /* istanbul ignore else: IE lt 9 */
        if (target.addEventListener) {
            target.addEventListener(eventName, func, false);
        } else {
            target.attachEvent('on' + eventName, func);
        }
    };

    var offEvent = function (target, eventName, func) {
        /* istanbul ignore else: IE lt 9 */
        if (target.removeEventListener) {
            target.removeEventListener(eventName, func, false);
        } else {
            target.detachEvent('on' + eventName, func);
        }
    };

    var onceEvent = function (target, eventName, func) {
        var callback = function () {
            offEvent(target, eventName, callback);
            return func.apply(this, arguments)
        };
        onEvent(target, eventName, callback);
    };

    var onceReadyStateChange = /* istanbul ignore next: IE lt 9 */ function (target, regex, func) {
        var eventName = 'readystatechange';
        var callback = function () {
            if (regex.test(target.readyState)) {
                offEvent(target, eventName, callback);
                return func.apply(this, arguments)
            }
        };
        onEvent(target, eventName, callback);
    };

    var createElementInDocument = function (document) {
        return function (tag, props, children) {
            var el = document.createElement(tag);
            if (props != null) {
                for (var prop in props) {
                    var val = props[prop];
                    if (val != null) {
                        if (el[prop] != null) {
                            el[prop] = val;
                        } else {
                            el.setAttribute(prop, val);
                        }
                    }
                }
            }
            if (children != null) {
                for (var i = 0, len = children.length; i < len; i++) {
                    var child = children[i];
                    el.appendChild(typeof child === 'string' ? document.createTextNode(child) : child);
                }
            }
            return el
        }
    };

    var createElement = createElementInDocument(document);

    var dispatchOnce = function (func) {
        var onceToken;
        return function () {
            if (!onceToken) {
                onceToken = 1;
                func.apply(this, arguments);
            }
        }
    };

    var defer = function (func) {
        /* istanbul ignore else */
        if (document.readyState === 'complete' || /* istanbul ignore next: IE lt 11 */ (document.readyState !== 'loading' && !document.documentElement.doScroll)) {
            setTimeout(func);
        } else {
            if (document.addEventListener) {
                var callback = dispatchOnce(func);
                onceEvent(document, 'DOMContentLoaded', callback);
                onceEvent(window, 'load', callback);
            } else {
                onceReadyStateChange(document, /m/, func);
            }
        }
    };

    var queues = {};

    var fetch = function (url, func) {
        var queue = queues[url] || (queues[url] = []);
        if (queue.push(func) > 1) {
            return
        }

        var callback = dispatchOnce(function () {
            delete queues[url];
            while ((func = queue.shift())) {
                func.apply(null, arguments);
            }
        });

        if (useXHR) {
            var xhr = new XMLHttpRequest();
            onEvent(xhr, 'abort', callback);
            onEvent(xhr, 'error', callback);
            onEvent(xhr, 'load', function () {
                var data;
                try {
                    data = JSON.parse(this.responseText);
                } catch (error) {
                    callback(error);
                    return
                }
                callback(this.status !== 200, data);
            });
            xhr.open('GET', url);
            xhr.send();
        } else {
            var contentWindow = this || window;
            contentWindow._ = function (json) {
                contentWindow._ = null;
                callback(json.meta.status !== 200, json.data);
            };
            var script = createElementInDocument(contentWindow.document)('script', {
                async: true,
                src: url + (url.indexOf('?') !== -1 ? '&' : '?') + 'callback=_'
            });
            var onloadend = /* istanbul ignore next: IE lt 9 */ function () {
                if (contentWindow._) {
                    contentWindow._({
                        meta: {}
                    });
                }
            };
            onEvent(script, 'load', onloadend);
            onEvent(script, 'error', onloadend);
            /* istanbul ignore if: IE lt 9 */
            if (script.readyState) {
                onceReadyStateChange(script, /de|m/, onloadend);
            }
            contentWindow.document.getElementsByTagName('head')[0].appendChild(script);
        }
    };

    var render = function (root, options, func) {
        var createElement = createElementInDocument(root.ownerDocument);

        var widget = createElement('ul', {className: 'github-facts'});
        root.appendChild(widget);

        var hostname = options.hostname.replace(/\.$/, '');
        if (hostname.length < domain.length || ('.' + hostname).substring(hostname.length - domain.length) !== ('.' + domain)) {
            options.removeAttribute('href');
            func(root);
            return
        }

        var path = (' /' + options.pathname).split(/\/+/);
        if (((hostname === domain || hostname === 'gist.' + domain) && path[3] === 'archive') ||
            (hostname === domain && path[3] === 'releases' && (path[4] === 'download' || (path[4] === 'latest' && path[5] === 'download'))) ||
            (hostname === 'codeload.' + domain)) {
            options.target = '_top';
        }

        var api = path[2] ? '/repos/' + path[1] + '/' + path[2] : '/users/' + path[1];
        fetch.call(this, apiBaseURL + api, function (error, json) {
            if (!error) {
                var stargazers_count = json['stargazers_count'];
                widget.appendChild(createElement('li', {
                    className: 'github-fact'
                }, [
                    ('' + stargazers_count).replace(/\B(?=(\d{3})+(?!\d))/g, ',') + ' Stars'
                ]));

                var forks_count = json['forks_count'];
                widget.appendChild(createElement('li', {
                    className: 'github-fact'
                }, [
                    ('' + forks_count).replace(/\B(?=(\d{3})+(?!\d))/g, ',') + ' Forks'
                ]));
            }
            func(root);
        });
    };

    var devicePixelRatio = window.devicePixelRatio || /* istanbul ignore next */ 1;

    var ceilPixel = function (px) {
        return (devicePixelRatio > 1 ? Math.ceil(Math.round(px * devicePixelRatio) / devicePixelRatio * 2) / 2 : Math.ceil(px)) || 0
    };

    var get = function (el) {
        var width = el.offsetWidth;
        var height = el.offsetHeight;
        if (el.getBoundingClientRect) {
            var boundingClientRect = el.getBoundingClientRect();
            width = Math.max(width, ceilPixel(boundingClientRect.width));
            height = Math.max(height, ceilPixel(boundingClientRect.height));
        }
        return [width, height]
    };

    var set = function (el, size) {
        el.style.width = size[0] + 'px';
        el.style.height = size[1] + 'px';
    };

    var render$1 = function (options, func) {
        if (options == null || func == null) {
            return
        }

        if (useShadowDOM) {
            var host = createElement('div', {className: 'github-repository'});
            render(host, options, function () {
                func(host);
            });
        } else {
            var iframe = createElement('iframe', {
                src: 'javascript:0',
                title: options.title || undefined,
                allowtransparency: true,
                scrolling: 'no',
                frameBorder: 0
            });
            set(iframe, [0, 0]);
            iframe.style.border = 'none';
            var callback = function () {
                var contentWindow = iframe.contentWindow;
                var body;
                try {
                    body = contentWindow.document.body;
                } catch (_) /* istanbul ignore next: IE 11 */ {
                    document.body.appendChild(iframe.parentNode.removeChild(iframe));
                    return
                }
                offEvent(iframe, 'load', callback);
                render.call(contentWindow, body, options, function (widget) {
                    var size = get(widget);
                    iframe.parentNode.removeChild(iframe);
                    onceEvent(iframe, 'load', function () {
                        set(iframe, size);
                    });
                    iframe.src = iframeURL + '#' + (iframe.name = stringify(options));
                    func(iframe);
                });
            };
            onEvent(iframe, 'load', callback);
            document.body.appendChild(iframe);
        }
    };

    if (location.protocol + '//' + location.host + location.pathname === iframeURL) {
        render(document.body, parse(window.name || location.hash.replace(/^#/, '')), function () { });
    } else {
        defer(function () {
            var ref = document.querySelectorAll
                ? document.querySelectorAll('a.' + buttonClass)
                : (function () {
                    var results = [];
                    var ref = document.getElementsByTagName('a');
                    for (var i = 0, len = ref.length; i < len; i++) {
                        if ((' ' + ref[i].className + ' ').replace(/[ \t\n\f\r]+/g, ' ').indexOf(' ' + buttonClass + ' ') !== -1) {
                            results.push(ref[i]);
                        }
                    }
                    return results
                })();
            for (var i = 0, len = ref.length; i < len; i++) {
                (function (anchor) {
                    render$1(anchor, function (el) {
                        anchor.appendChild(el);
                    });
                })(ref[i]);
            }
        });
    }

}());
